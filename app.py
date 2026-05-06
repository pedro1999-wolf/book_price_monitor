"""
Captador O'Reilly — Dashboard de preços de livros.

Mostra:
- Melhor plataforma para comprar HOJE (por livro)
- Variação % em relação ao dia anterior
- Histórico mensal de preço (apenas Data Science do Zero, fonte: historico_precos_amazon)
- Previsão mensal de preço (apenas Data Science do Zero, fonte: previsoes_preco)
  com aviso de baixa confiança quando os valores saem da faixa razoável.

Roda contra Supabase (projeto wsvcmcixojolpfnnggfl). Configure SUPABASE_URL e
SUPABASE_KEY como secrets/env vars (use a anon ou service-role; anon basta se as
tabelas tiverem policies de leitura).
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from supabase import Client, create_client

# ────────────────────────────── CONFIG ──────────────────────────────

LIVROS = {
    "data_science_do_zero_2ed": "Data Science do Zero (2ª ed)",
    "maos_a_obra_ml_sklearn_keras_tensorflow": "Mãos à Obra: ML com Scikit-Learn, Keras e TensorFlow",
}

# Apenas este livro tem histórico mensal e previsão por enquanto.
LIVRO_COM_HISTORICO = "data_science_do_zero_2ed"

# Faixa razoável para sinalizar previsões suspeitas.
# Histórico observado fica entre ~R$60 e ~R$210 (pico Mar/2023).
PREVISAO_MIN_RAZOAVEL = 30.0
PREVISAO_MAX_RAZOAVEL = 250.0

st.set_page_config(
    page_title="Captador O'Reilly — Preços",
    page_icon="📚",
    layout="wide",
)


# ────────────────────────────── SUPABASE ──────────────────────────────

@st.cache_resource
def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        st.error(
            "Faltam credenciais. Defina SUPABASE_URL e SUPABASE_KEY em st.secrets "
            "ou variáveis de ambiente."
        )
        st.stop()
    return create_client(url, key)


@st.cache_data(ttl=300)
def load_platform_snapshots(book_key: str) -> pd.DataFrame:
    sb = get_supabase()
    res = (
        sb.table("book_platform_price_snapshots")
        .select("run_date, platform, found_match, best_price, currency, source, product_url")
        .eq("book_key", book_key)
        .order("run_date", desc=True)
        .limit(2000)
        .execute()
    )
    df = pd.DataFrame(res.data)
    if df.empty:
        return df
    df["run_date"] = pd.to_datetime(df["run_date"]).dt.date
    df["best_price"] = pd.to_numeric(df["best_price"], errors="coerce")
    return df


@st.cache_data(ttl=300)
def load_daily_best(book_key: str) -> pd.DataFrame:
    sb = get_supabase()
    res = (
        sb.table("book_daily_best_snapshots")
        .select("run_date, found_match, best_platform, best_price, currency, product_url")
        .eq("book_key", book_key)
        .order("run_date", desc=True)
        .limit(365)
        .execute()
    )
    df = pd.DataFrame(res.data)
    if df.empty:
        return df
    df["run_date"] = pd.to_datetime(df["run_date"]).dt.date
    df["best_price"] = pd.to_numeric(df["best_price"], errors="coerce")
    return df


@st.cache_data(ttl=300)
def load_historico() -> pd.DataFrame:
    sb = get_supabase()
    res = (
        sb.table("historico_precos_amazon")
        .select("periodo, data_label, preco_buy_box, preco_amazon")
        .order("periodo")
        .execute()
    )
    df = pd.DataFrame(res.data)
    if df.empty:
        return df
    df["preco_buy_box"] = pd.to_numeric(df["preco_buy_box"], errors="coerce")
    df["preco_amazon"] = pd.to_numeric(df["preco_amazon"], errors="coerce")
    df["dt"] = pd.to_datetime(df["periodo"] + "-01")
    return df


@st.cache_data(ttl=300)
def load_previsoes() -> pd.DataFrame:
    sb = get_supabase()
    res = (
        sb.table("previsoes_preco")
        .select("*")
        .order("periodo")
        .execute()
    )
    df = pd.DataFrame(res.data)
    if df.empty:
        return df
    for col in ("preco_previsto", "intervalo_min", "intervalo_max", "threshold_usado"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dt"] = pd.to_datetime(df["periodo"] + "-01")
    return df


# ────────────────────────────── HELPERS ──────────────────────────────

def fmt_brl(v: Optional[float]) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct_delta(novo: Optional[float], antigo: Optional[float]) -> Optional[float]:
    if novo is None or antigo is None or pd.isna(novo) or pd.isna(antigo) or antigo == 0:
        return None
    return (novo - antigo) / antigo * 100.0


def previsao_suspeita(row) -> bool:
    pp = row.get("preco_previsto")
    if pp is None or pd.isna(pp):
        return True
    if pp < PREVISAO_MIN_RAZOAVEL or pp > PREVISAO_MAX_RAZOAVEL:
        return True
    mn, mx = row.get("intervalo_min"), row.get("intervalo_max")
    if pd.notna(mn) and mn < 0:
        return True
    return False


# ────────────────────────────── UI ──────────────────────────────

st.title("📚 Captador O'Reilly — Preços de livros")
st.caption("Dados atualizados diariamente via GitHub Actions · Fonte: Supabase")

with st.sidebar:
    st.header("Filtros")
    book_key = st.selectbox(
        "Livro",
        options=list(LIVROS.keys()),
        format_func=lambda k: LIVROS[k],
    )
    st.markdown("---")
    st.caption("ℹ️ Histórico e previsão estão disponíveis apenas para *Data Science do Zero* por enquanto.")

# ─────────── Bloco 1: melhor plataforma hoje + delta vs ontem ───────────

plat = load_platform_snapshots(book_key)
daily_best = load_daily_best(book_key)

if plat.empty:
    st.warning("Sem dados de plataforma para este livro ainda.")
    st.stop()

datas_disponiveis = sorted(plat["run_date"].unique(), reverse=True)
hoje = datas_disponiveis[0]
ontem = datas_disponiveis[1] if len(datas_disponiveis) > 1 else None

plat_hoje = plat[plat["run_date"] == hoje].copy()
plat_ontem = plat[plat["run_date"] == ontem].copy() if ontem else pd.DataFrame()

# best platform hoje (a partir do daily_best, que já tem a lógica oficial)
best_today_row = daily_best[daily_best["run_date"] == hoje]
if not best_today_row.empty and best_today_row.iloc[0]["found_match"]:
    best_platform = best_today_row.iloc[0]["best_platform"]
    best_price_today = best_today_row.iloc[0]["best_price"]
    best_url = best_today_row.iloc[0].get("product_url")
else:
    # fallback: pega o menor preço entre as plataformas hoje
    achados = plat_hoje[plat_hoje["found_match"] & plat_hoje["best_price"].notna()]
    if achados.empty:
        best_platform, best_price_today, best_url = None, None, None
    else:
        idx = achados["best_price"].idxmin()
        best_platform = achados.loc[idx, "platform"]
        best_price_today = achados.loc[idx, "best_price"]
        best_url = achados.loc[idx, "product_url"]

# preço do mesmo livro na mesma plataforma ontem (para delta)
best_price_yesterday = None
if best_platform and ontem is not None:
    match = plat_ontem[
        (plat_ontem["platform"] == best_platform) & (plat_ontem["found_match"])
    ]
    if not match.empty:
        best_price_yesterday = match.iloc[0]["best_price"]

st.subheader(f"Hoje — {hoje.strftime('%d/%m/%Y')}")
col1, col2, col3 = st.columns(3)
col1.metric("Melhor plataforma", best_platform or "—")
delta_str = None
if best_price_today is not None and best_price_yesterday is not None:
    d = pct_delta(best_price_today, best_price_yesterday)
    if d is not None:
        delta_str = f"{d:+.2f}% vs ontem"
col2.metric(
    "Melhor preço",
    fmt_brl(best_price_today),
    delta=delta_str,
    delta_color="inverse",  # subir é ruim para o comprador
)
if best_url:
    col3.markdown(f"**Link**\n\n[Abrir oferta]({best_url})")

# ─────────── Bloco 2: tabela comparativa por plataforma ───────────

st.subheader("Preços por plataforma")
plat_hoje_view = plat_hoje[["platform", "found_match", "best_price", "source", "product_url"]].rename(
    columns={
        "platform": "Plataforma",
        "found_match": "Encontrado",
        "best_price": "Preço hoje",
        "source": "Fonte",
        "product_url": "URL",
    }
)
# adiciona Δ vs ontem por plataforma
if not plat_ontem.empty:
    ont_lookup = plat_ontem.set_index("platform")["best_price"].to_dict()
    plat_hoje_view["Preço ontem"] = plat_hoje_view["Plataforma"].map(ont_lookup)
    plat_hoje_view["Δ %"] = plat_hoje_view.apply(
        lambda r: pct_delta(r["Preço hoje"], r["Preço ontem"]), axis=1
    )
else:
    plat_hoje_view["Preço ontem"] = None
    plat_hoje_view["Δ %"] = None

plat_hoje_view = plat_hoje_view[
    ["Plataforma", "Encontrado", "Preço hoje", "Preço ontem", "Δ %", "Fonte", "URL"]
].sort_values("Preço hoje", na_position="last")

st.dataframe(
    plat_hoje_view,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Preço hoje": st.column_config.NumberColumn(format="R$ %.2f"),
        "Preço ontem": st.column_config.NumberColumn(format="R$ %.2f"),
        "Δ %": st.column_config.NumberColumn(format="%.2f%%"),
        "URL": st.column_config.LinkColumn("URL", display_text="abrir"),
    },
)

# ─────────── Bloco 3 & 4: histórico + previsão (só DSdZ) ───────────

if book_key != LIVRO_COM_HISTORICO:
    st.info(
        f"Histórico mensal e previsão estão disponíveis apenas para "
        f"*{LIVROS[LIVRO_COM_HISTORICO]}* por enquanto."
    )
else:
    hist = load_historico()
    prev = load_previsoes()

    st.subheader("Histórico mensal de preço (Buy Box / Amazon)")
    if hist.empty:
        st.warning("Sem histórico ainda.")
    else:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Scatter(
                x=hist["dt"],
                y=hist["preco_buy_box"],
                name="Buy Box",
                mode="lines+markers",
                line=dict(width=2),
            )
        )
        if hist["preco_amazon"].notna().any():
            fig_hist.add_trace(
                go.Scatter(
                    x=hist["dt"],
                    y=hist["preco_amazon"],
                    name="Amazon (direto)",
                    mode="lines+markers",
                    line=dict(width=2, dash="dot"),
                )
            )
        fig_hist.update_layout(
            xaxis_title="Mês",
            yaxis_title="Preço (R$)",
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Previsão mensal")
    if prev.empty:
        st.warning("Sem previsões geradas ainda.")
    else:
        prev = prev.copy()
        prev["suspeita"] = prev.apply(previsao_suspeita, axis=1)

        suspeitas = prev[prev["suspeita"]]
        if not suspeitas.empty:
            periodos = ", ".join(suspeitas["periodo"].astype(str).tolist())
            st.warning(
                f"⚠️ Baixa confiança: as previsões para **{periodos}** estão "
                f"fora da faixa esperada (R${PREVISAO_MIN_RAZOAVEL:.0f}–"
                f"R${PREVISAO_MAX_RAZOAVEL:.0f}) ou com intervalo negativo. "
                "O modelo provavelmente precisa ser revisitado — verifique se "
                "há transformação log/diff sem inversão correta, ou re-treine "
                "com mais histórico."
            )

        fig_prev = go.Figure()
        # último ponto histórico para conectar
        if not hist.empty:
            ultimo = hist.dropna(subset=["preco_buy_box"]).tail(6)
            fig_prev.add_trace(
                go.Scatter(
                    x=ultimo["dt"],
                    y=ultimo["preco_buy_box"],
                    name="Histórico recente",
                    mode="lines+markers",
                    line=dict(width=2, color="#888"),
                )
            )
        # banda de confiança
        fig_prev.add_trace(
            go.Scatter(
                x=list(prev["dt"]) + list(prev["dt"][::-1]),
                y=list(prev["intervalo_max"]) + list(prev["intervalo_min"][::-1]),
                fill="toself",
                fillcolor="rgba(99,110,250,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Intervalo",
                hoverinfo="skip",
            )
        )
        fig_prev.add_trace(
            go.Scatter(
                x=prev["dt"],
                y=prev["preco_previsto"],
                name="Previsto",
                mode="lines+markers",
                line=dict(width=2, color="#636EFA"),
            )
        )
        fig_prev.update_layout(
            xaxis_title="Mês",
            yaxis_title="Preço (R$)",
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig_prev, use_container_width=True)

        st.dataframe(
            prev[["periodo", "preco_previsto", "intervalo_min", "intervalo_max",
                  "abaixo_threshold", "threshold_usado", "gerado_em", "suspeita"]]
            .rename(columns={
                "periodo": "Mês",
                "preco_previsto": "Previsto",
                "intervalo_min": "Min",
                "intervalo_max": "Max",
                "abaixo_threshold": "Abaixo do threshold",
                "threshold_usado": "Threshold",
                "gerado_em": "Gerado em",
                "suspeita": "Baixa confiança",
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Previsto": st.column_config.NumberColumn(format="R$ %.2f"),
                "Min": st.column_config.NumberColumn(format="R$ %.2f"),
                "Max": st.column_config.NumberColumn(format="R$ %.2f"),
                "Threshold": st.column_config.NumberColumn(format="R$ %.2f"),
            },
        )
