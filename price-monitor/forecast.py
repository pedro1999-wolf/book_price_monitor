"""
forecast.py
-----------
Busca histórico de preços do Supabase, cruza com câmbio USD/BRL (BCB/PTAX),
infere estoque e roda Prophet para prever preço nos próximos 3 meses.
Salva previsão no Supabase — consultada pelo app Streamlit.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from prophet import Prophet
from supabase import create_client

SUPABASE_URL    = os.environ["SUPABASE_URL"]
SUPABASE_KEY    = os.environ["SUPABASE_KEY"]
PRICE_THRESHOLD = float(os.environ.get("PRICE_THRESHOLD", "75.00"))
FORECAST_DAYS   = 90

sb = create_client(SUPABASE_URL, SUPABASE_KEY)


def load_price_history() -> pd.DataFrame:
    resp = sb.table("historico_precos_amazon") \
             .select("periodo, preco_buy_box, preco_amazon") \
             .order("periodo") \
             .execute()

    df = pd.DataFrame(resp.data)
    df["ds"] = pd.to_datetime(df["periodo"] + "-01")
    df["y"]  = df["preco_buy_box"].astype(float)
    df["amazon_presente"] = df["preco_amazon"].notna().astype(int)
    df["sem_estoque"] = (
        df["preco_amazon"].isna() & (df["preco_buy_box"] > df["preco_buy_box"].quantile(0.6))
    ).astype(int)
    return df[["ds", "y", "amazon_presente", "sem_estoque"]]


def load_exchange_rate(start: str, end: str) -> pd.DataFrame:
    url = (
        "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
        "CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
        "?@dataInicial='{start}'&@dataFinalCotacao='{end}'"
        "&$top=10000&$format=json&$select=cotacaoVenda,dataHoraCotacao"
    ).format(start=start, end=end)

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()["value"]
        df = pd.DataFrame(data)
        df["dataHoraCotacao"] = pd.to_datetime(df["dataHoraCotacao"])
        df["mes"] = df["dataHoraCotacao"].dt.to_period("M").dt.to_timestamp()
        monthly = df.groupby("mes")["cotacaoVenda"].mean().reset_index()
        monthly.columns = ["ds", "usd_brl"]
        return monthly
    except Exception as e:
        print(f"[AVISO] BCB falhou: {e}. Usando yfinance como fallback.")
        return load_exchange_rate_yahoo()


def load_exchange_rate_yahoo() -> pd.DataFrame:
    import yfinance as yf
    hist = yf.Ticker("BRL=X").history(period="5y", interval="1mo")
    hist = hist.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "usd_brl"})
    hist["ds"] = pd.to_datetime(hist["ds"]).dt.to_period("M").dt.to_timestamp()
    return hist


def forecast_exchange_rate(fx_df: pd.DataFrame, days: int) -> pd.DataFrame:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(fx_df.rename(columns={"usd_brl": "y"}))
    future = m.make_future_dataframe(periods=days, freq="D")
    fc = m.predict(future)[["ds", "yhat"]].rename(columns={"yhat": "usd_brl_forecast"})
    fc["ds"] = fc["ds"].dt.to_period("M").dt.to_timestamp()
    return fc.groupby("ds")["usd_brl_forecast"].mean().reset_index()


def run_forecast(price_df: pd.DataFrame, fx_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.merge(fx_df, on="ds", how="left")
    df["usd_brl"] = df["usd_brl"].interpolate(method="linear").bfill()

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,
        seasonality_prior_scale=10,
        interval_width=0.90,
    )
    m.add_regressor("usd_brl",         standardize=True)
    m.add_regressor("sem_estoque",     standardize=False)
    m.add_regressor("amazon_presente", standardize=False)
    m.fit(df)

    future = m.make_future_dataframe(periods=FORECAST_DAYS, freq="D")
    fx_future = forecast_exchange_rate(fx_df, FORECAST_DAYS)
    future = future.merge(fx_future, on="ds", how="left")
    future["usd_brl"]          = future["usd_brl_forecast"].fillna(df["usd_brl"].iloc[-1])
    future["sem_estoque"]      = 0
    future["amazon_presente"]  = 1

    forecast = m.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def save_forecast(forecast: pd.DataFrame):
    future_only = forecast[forecast["ds"] > datetime.now()].copy()
    future_only["periodo"] = future_only["ds"].dt.to_period("M").dt.strftime("%Y-%m")

    monthly = future_only.groupby("periodo").agg(
        preco_previsto=("yhat",       "mean"),
        intervalo_min =("yhat_lower", "mean"),
        intervalo_max =("yhat_upper", "mean"),
    ).reset_index()

    monthly["abaixo_threshold"] = monthly["preco_previsto"] <= PRICE_THRESHOLD
    monthly["threshold_usado"]  = PRICE_THRESHOLD

    records = monthly.to_dict(orient="records")

    sb.table("previsoes_preco").delete().neq("id", 0).execute()
    sb.table("previsoes_preco").insert(records).execute()

    n_alertas = monthly["abaixo_threshold"].sum()
    print(f"✅ {len(records)} previsões salvas no Supabase.")
    print(f"⚠️  {n_alertas} mês(es) com preço previsto abaixo de R$ {PRICE_THRESHOLD:.2f}")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    print("🔄 Carregando histórico de preços...")
    price_df = load_price_history()

    print("🔄 Buscando câmbio USD/BRL (BCB/PTAX)...")
    start_date = price_df["ds"].min().strftime("%m-%d-%Y")
    end_date   = datetime.now().strftime("%m-%d-%Y")
    fx_df      = load_exchange_rate(start_date, end_date)

    print("🔄 Rodando modelo Prophet...")
    forecast = run_forecast(price_df, fx_df)

    print("💾 Salvando previsão no Supabase...")
    save_forecast(forecast)
