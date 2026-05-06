"""
Fechamento mensal do histórico Amazon do livro Data Science do Zero.

Lógica:
1. Determina o mês a fechar:
   - Se for chamado sem argumento, fecha o MÊS ANTERIOR ao corrente.
   - Aceita --periodo YYYY-MM para forçar um período específico.
2. Calcula a média de best_price da plataforma 'Amazon' em
   book_platform_price_snapshots para o livro data_science_do_zero_2ed
   no período (apenas linhas com found_match=true e best_price not null).
3. Faz UPSERT em historico_precos_amazon (chave única: periodo).
4. Dispara o re-treino do modelo (retrain_forecast.retrain) que regrava
   as previsões em previsoes_preco.

Uso (GitHub Actions, dia 1 de cada mês):
    python monthly_close.py

Uso manual:
    python monthly_close.py --periodo 2026-04
    python monthly_close.py --livro data_science_do_zero_2ed
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from typing import Optional

from supabase import Client, create_client

LIVRO_DEFAULT = "data_science_do_zero_2ed"
PLATAFORMA_AMAZON = "Amazon"
TABELA_SNAPSHOTS = "book_platform_price_snapshots"
TABELA_HISTORICO = "historico_precos_amazon"
FONTE = "Média mensal Amazon (book_platform_price_snapshots)"

MESES_PT = [
    "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
    "Jul", "Ago", "Set", "Out", "Nov", "Dez",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("monthly_close")


def get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    # Use service-role aqui — o job precisa escrever.
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def previous_month(today: date) -> str:
    if today.month == 1:
        return f"{today.year - 1}-12"
    return f"{today.year}-{today.month - 1:02d}"


def data_label_for(periodo: str) -> str:
    ano, mes = periodo.split("-")
    return f"{MESES_PT[int(mes) - 1]} {ano}"


def fetch_amazon_avg(sb: Client, livro: str, periodo: str) -> Optional[float]:
    """Média do best_price da Amazon para o livro no período (YYYY-MM)."""
    inicio = f"{periodo}-01"
    ano, mes = map(int, periodo.split("-"))
    if mes == 12:
        fim = f"{ano + 1}-01-01"
    else:
        fim = f"{ano}-{mes + 1:02d}-01"

    res = (
        sb.table(TABELA_SNAPSHOTS)
        .select("best_price")
        .eq("book_key", livro)
        .eq("platform", PLATAFORMA_AMAZON)
        .eq("found_match", True)
        .gte("run_date", inicio)
        .lt("run_date", fim)
        .execute()
    )
    precos = [
        float(row["best_price"])
        for row in res.data
        if row.get("best_price") is not None
    ]
    if not precos:
        return None
    return sum(precos) / len(precos)


def upsert_historico(sb: Client, periodo: str, preco: float) -> dict:
    payload = {
        "periodo": periodo,
        "data_label": data_label_for(periodo),
        "preco_buy_box": round(preco, 2),
        "preco_amazon": round(preco, 2),  # mesma fonte por enquanto
        "fonte": FONTE,
    }
    # on_conflict pelo índice único de `periodo`.
    res = (
        sb.table(TABELA_HISTORICO)
        .upsert(payload, on_conflict="periodo")
        .execute()
    )
    return res.data


def trigger_retrain() -> None:
    """Chama o re-treino do modelo, se o módulo existir.

    Mantemos isso opcional pra que o monthly_close seja útil mesmo antes do
    pipeline de modelo estar versionado neste repo.
    """
    try:
        from retrain_forecast import retrain  # type: ignore
    except ImportError:
        log.warning(
            "Módulo retrain_forecast não encontrado — pulando re-treino. "
            "Adicione retrain_forecast.py no repo para encadear o re-treino."
        )
        return
    log.info("Disparando re-treino do modelo de previsão…")
    retrain()
    log.info("Re-treino concluído.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--periodo",
        help="Período YYYY-MM a fechar. Default: mês anterior ao atual.",
    )
    parser.add_argument(
        "--livro",
        default=LIVRO_DEFAULT,
        help=f"book_key. Default: {LIVRO_DEFAULT}.",
    )
    parser.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Não dispara o re-treino do modelo.",
    )
    args = parser.parse_args()

    periodo = args.periodo or previous_month(date.today())
    livro = args.livro

    log.info("Fechando período %s para livro %s", periodo, livro)
    sb = get_client()

    media = fetch_amazon_avg(sb, livro, periodo)
    if media is None:
        log.error(
            "Sem dados Amazon para %s em %s — nada a inserir. "
            "Verifique se o scraper rodou no período.",
            livro, periodo,
        )
        return 2

    log.info("Média Amazon de %s em %s: R$ %.2f", livro, periodo, media)
    out = upsert_historico(sb, periodo, media)
    log.info("Upsert em %s OK (%d linha(s)).", TABELA_HISTORICO, len(out or []))

    if not args.skip_retrain:
        trigger_retrain()

    return 0


if __name__ == "__main__":
    sys.exit(main())
