"""
Re-treino do modelo de previsão mensal (Data Science do Zero).

Substitui o antigo price-monitor/forecast.py (Prophet com inferência de
estoque) que produzia previsões anômalas (ex.: R$ -53,98 para Mai/2026).

Pipeline:
1. Lê historico_precos_amazon (preco_buy_box).
2. Trata outliers (winsorize por IQR) — picos de stock-out distorcem o modelo.
3. Transforma com log para garantir previsões > 0.
4. Modela com Holt-Winters (Exponential Smoothing) — robusto e
   apropriado para séries mensais com tendência leve.
5. Inverte a transformação (exp), gera intervalos de confiança lognormais.
6. Substitui as previsões em previsoes_preco para os próximos N meses.

Por que NÃO usamos mais Prophet + 'sem_estoque'?
O regressor 'sem_estoque' era inferido como (preco_amazon NULL) AND
(preco_buy_box > Q60). Isso marca quase todos os picos de stock-out como
sem_estoque=1. O modelo aprendia um coeficiente alto pra essa feature, e
no momento da previsão a feature era forçada para 0, removendo um boost
aprendido — o que podia jogar a previsão para valores negativos.

Uso:
    python retrain_forecast.py                  # 3 meses à frente
    python retrain_forecast.py --horizon 6      # 6 meses
    python retrain_forecast.py --threshold 70   # threshold de "abaixo do alvo"
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List

import numpy as np
import pandas as pd
from supabase import Client, create_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("retrain_forecast")

TABELA_HISTORICO = "historico_precos_amazon"
TABELA_PREVISOES = "previsoes_preco"
DEFAULT_HORIZON = 3
DEFAULT_THRESHOLD = 75.0

# Constante z para intervalo ~95% (assumindo resíduos aprox normais em log).
Z95 = 1.96


def get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def load_history(sb: Client) -> pd.DataFrame:
    res = (
        sb.table(TABELA_HISTORICO)
        .select("periodo, preco_buy_box")
        .order("periodo")
        .execute()
    )
    df = pd.DataFrame(res.data)
    df["preco_buy_box"] = pd.to_numeric(df["preco_buy_box"], errors="coerce")
    df = df.dropna(subset=["preco_buy_box"])
    df["dt"] = pd.to_datetime(df["periodo"] + "-01")
    df = df.set_index("dt").sort_index()
    return df


def winsorize_iqr(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Cap valores fora de [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    capped = s.clip(lower=lo, upper=hi)
    n_clipped = (s != capped).sum()
    if n_clipped:
        log.info("Winsorize: %d valor(es) ajustado(s) (lo=%.2f, hi=%.2f).",
                 n_clipped, lo, hi)
    return capped


def fit_and_forecast(y: pd.Series, horizon: int) -> pd.DataFrame:
    """
    Ajusta Holt-Winters em log(y) e devolve DataFrame com:
    periodo, preco_previsto, intervalo_min, intervalo_max.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    log_y = np.log(y)

    # Tendência aditiva, sem sazonalidade fixa. Para séries com sazonalidade
    # anual nítida, troque para seasonal="add", seasonal_periods=12.
    model = ExponentialSmoothing(
        log_y,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)

    # Resíduos no espaço log → desvio padrão para banda de confiança.
    resid = log_y - fit.fittedvalues
    sigma = resid.std(ddof=1)

    forecast_log = fit.forecast(horizon)

    # Datas dos próximos `horizon` meses
    last = y.index[-1]
    future_idx = pd.date_range(
        start=last + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    # Inverte log e calcula bandas — em log a banda é simétrica;
    # em preço, vira lognormal (assimétrica e sempre positiva).
    pred = np.exp(forecast_log.values)
    lo = np.exp(forecast_log.values - Z95 * sigma)
    hi = np.exp(forecast_log.values + Z95 * sigma)

    out = pd.DataFrame({
        "periodo": [d.strftime("%Y-%m") for d in future_idx],
        "preco_previsto": np.round(pred, 2),
        "intervalo_min": np.round(lo, 2),
        "intervalo_max": np.round(hi, 2),
    })
    return out


def write_predictions(sb: Client, preds: pd.DataFrame, threshold: float) -> int:
    """Substitui completamente as previsões: apaga as antigas e insere as novas."""
    sb.table(TABELA_PREVISOES).delete().neq("id", -1).execute()

    rows: List[dict] = []
    for _, r in preds.iterrows():
        rows.append({
            "periodo": r["periodo"],
            "preco_previsto": float(r["preco_previsto"]),
            "intervalo_min": float(r["intervalo_min"]),
            "intervalo_max": float(r["intervalo_max"]),
            "abaixo_threshold": bool(r["preco_previsto"] < threshold),
            "threshold_usado": float(threshold),
        })
    sb.table(TABELA_PREVISOES).insert(rows).execute()
    return len(rows)


def retrain(horizon: int = DEFAULT_HORIZON,
            threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Função pública — também usada pelo monthly_close.trigger_retrain()."""
    sb = get_client()
    log.info("Carregando histórico…")
    df = load_history(sb)
    if len(df) < 12:
        raise RuntimeError(
            f"Histórico curto demais ({len(df)} meses) — precisa de pelo menos 12."
        )
    log.info("Histórico: %d meses (de %s a %s).",
             len(df), df.index[0].date(), df.index[-1].date())

    y = winsorize_iqr(df["preco_buy_box"])
    preds = fit_and_forecast(y, horizon=horizon)
    log.info("Previsões geradas:\n%s", preds.to_string(index=False))

    n = write_predictions(sb, preds, threshold)
    log.info("Gravadas %d previsões em %s.", n, TABELA_PREVISOES)
    return preds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    retrain(horizon=args.horizon, threshold=args.threshold)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
