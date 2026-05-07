"""
Captador O'Reilly — Dashboard de preços de livros.

Identidade visual inspirada nos livros da O'Reilly: fundo creme, tipografia
serifada para corpo, sans-serif condensada para títulos, acento vermelho
escuro, traços finos.

Bloco único de série temporal: histórico mensal e previsão emendados, com
banda de confiança e separador "hoje".
"""

from __future__ import annotations

import os
from datetime import date
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

LIVRO_COM_HISTORICO = "data_science_do_zero_2ed"

# Faixa razoável para sinalizar previsões suspeitas.
PREVISAO_MIN_RAZOAVEL = 30.0
PREVISAO_MAX_RAZOAVEL = 250.0

# Paleta O'Reilly
CREAM = "#F5F1E8"
PAPER = "#FAF7EC"
INK = "#1A1A1A"
INK_SOFT = "#4A4A4A"
RULE = "#D9D2BF"
RED = "#A41E22"
RED_SOFT = "rgba(164,30,34,0.15)"
BLUE = "#1F4E79"

st.set_page_config(
    page_title="Captador O'Reilly — Preços",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ────────────────────────────── ESTILO ──────────────────────────────

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600&family=Oswald:wght@500;700&display=swap');

      .stApp {{
        background-color: {CREAM};
      }}
      .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
      }}
      html, body, [class*="css"] {{
        font-family: "Crimson Pro", Georgia, serif;
        color: {INK};
      }}
      h1, h2, h3, h4, .oreilly-eyebrow {{
        font-family: "Oswald", "Helvetica Neue", Arial, sans-serif;
        letter-spacing: 0.04em;
        color: {INK};
      }}
      .oreilly-banner {{
        border-top: 2px solid {INK};
        border-bottom: 1px solid {INK};
        padding: 1.2rem 0;
        margin-bottom: 2rem;
        background: {PAPER};
      }}
      .oreilly-eyebrow {{
        font-size: 0.85rem;
        text-transform: uppercase;
        color: {RED};
        font-weight: 700;
      }}
      .oreilly-title {{
        font-family: "Oswald", sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.05;
        margin: 0.2rem 0 0.4rem 0;
      }}
      .oreilly-subtitle {{
        font-family: "Crimson Pro", serif;
        font-style: italic;
        color: {INK_SOFT};
        font-size: 1.05rem;
      }}
      .oreilly-section {{
        font-family: "Oswald", sans-serif;
        font-weight: 700;
        font-size: 1.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 2rem 0 0.6rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid {INK};
      }}
      .stMetric {{
        background: {PAPER};
        border: 1px solid {RULE};
        padding: 0.8rem 1rem;
        border-radius: 0;
      }}
      .stMetric label {{
        font-family: "Oswald", sans-serif !important;
        text-transform: uppercase;
        font-size: 0.78rem !important;
        color: {INK_SOFT} !important;
        letter-spacing: 0.06em;
      }}
      .stMetric [data-testid="stMetricValue"] {{
        font-family: "Oswald", sans-serif !important;
        font-size: 1.8rem !important;
        color: {INK} !important;
      }}
      .stDataFrame, .stTable {{
        background: {PAPER} !important;
        border: 1px solid {RULE} !important;
      }}
      a {{ color: {RED}; }}
      .oreilly-footer {{
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid {RULE};
        font-size: 0.85rem;
        color: {INK_SOFT};
        font-style: italic;
        text-align: center;
      }}
      [data-testid="stSidebar"] {{
        background-color: {PAPER};
        border-right: 1px solid {RULE};
      }}
      .oreilly-callout {{
        border-left: 3px solid {RED};
        background: {PAPER};
        padding: 0.8rem 1rem;
        margin: 1rem 0;
        font-family: "Crimson Pro", serif;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────── SUPABASE ──────────────────────────────

@st.cache_resource
def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        try:
            url = url or st.secrets.get("SUPABASE_URL")
            key = key or st.secrets.get("SUPABASE_KEY")
        except Exception:
            pass
    if not url or not key:
        st.error("Faltam credenciais. Defina SUPABASE_URL e SUPABASE_KEY em st.secrets ou env.")
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
    res = sb.table("previsoes_preco").select("*").order("periodo").execute()
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
    mn = row.get("intervalo_min")
    if pd.notna(mn) and mn < 0:
        return True
    return False


# ────────────────────────────── HEADER ──────────────────────────────

st.markdown(
    """
    <div class="oreilly-banner">
      <div class="oreilly-eyebrow">A Practical Field Guide</div>
      <div class="oreilly-title">Captador O'Reilly</div>
      <div class="oreilly-subtitle">
        Monitoramento diário de preços e previsão mensal — coletado e
        modelado todo dia via GitHub Actions, servido pelo Supabase.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="oreilly-eyebrow">Edição</div>', unsafe_allow_html=True)
    book_key = st.selectbox(
        "Livro",
        options=list(LIVROS.keys()),
        format_func=lambda k: LIVROS[k],
        label_visibility="collapsed",
    )
    st.caption("Histórico mensal e previsão estão disponíveis apenas para *Data Science do Zero* por enquanto.")

# ─────────── Bloco 1: melhor plataforma hoje ───────────

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

best_today_row = daily_best[daily_best["run_date"] == hoje]
if not best_today_row.empty and best_today_row.iloc[0]["found_match"]:
    best_platform = best_today_row.iloc[0]["best_platform"]
    best_price_today = best_today_row.iloc[0]["best_price"]
    best_url = best_today_row.iloc[0].get("product_url")
else:
    achados = plat_hoje[plat_hoje["found_match"] & plat_hoje["best_price"].notna()]
    if achados.empty:
        best_platform, best_price_today, best_url = None, None, None
    else:
        idx = achados["best_price"].idxmin()
        best_platform = achados.loc[idx, "platform"]
        best_price_today = achados.loc[idx, "best_price"]
        best_url = achados.loc[idx, "product_url"]

best_price_yesterday = None
if best_platform and ontem is not None:
    match = plat_ontem[
        (plat_ontem["platform"] == best_platform) & (plat_ontem["found_match"])
    ]
    if not match.empty:
        best_price_yesterday = match.iloc[0]["best_price"]

st.markdown(
    f'<div class="oreilly-section">Capítulo 1 — Hoje, {hoje.strftime("%d/%m/%Y")}</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1.2, 1.2, 1])
col1.metric("Melhor plataforma", best_platform or "—")
delta_str = None
if best_price_today is not None and best_price_yesterday is not None:
    d = pct_delta(best_price_today, best_price_yesterday)
    if d is not None:
        delta_str = f"{d:+.2f}% vs. ontem"
col2.metric(
    "Melhor preço",
    fmt_brl(best_price_today),
    delta=delta_str,
    delta_color="inverse",
)
if best_url:
    col3.markdown(
        f'<div class="oreilly-callout">'
        f'<strong>Onde comprar</strong><br>'
        f'<a href="{best_url}" target="_blank">→ Abrir oferta no {best_platform}</a>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─────────── Bloco 2: tabela ───────────

st.markdown(
    '<div class="oreilly-section">Capítulo 2 — Comparativo por plataforma</div>',
    unsafe_allow_html=True,
)

plat_hoje_view = plat_hoje[["platform", "found_match", "best_price", "source", "product_url"]].rename(
    columns={
        "platform": "Plataforma",
        "found_match": "Encontrado",
        "best_price": "Preço hoje",
        "source": "Fonte",
        "product_url": "URL",
    }
)
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
    width="stretch",
    hide_index=True,
    column_config={
        "Preço hoje": st.column_config.NumberColumn(format="R$ %.2f"),
        "Preço ontem": st.column_config.NumberColumn(format="R$ %.2f"),
        "Δ %": st.column_config.NumberColumn(format="%.2f%%"),
        "URL": st.column_config.LinkColumn("URL", display_text="abrir"),
    },
)

# ─────────── Bloco 3: série temporal unificada ───────────

if book_key != LIVRO_COM_HISTORICO:
    st.markdown(
        f'<div class="oreilly-callout">Histórico mensal e previsão estão disponíveis '
        f'apenas para <em>{LIVROS[LIVRO_COM_HISTORICO]}</em> por enquanto.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="oreilly-section">Capítulo 3 — Histórico & previsão</div>',
        unsafe_allow_html=True,
    )

    hist = load_historico().sort_values("dt")
    prev = load_previsoes().sort_values("dt")

    if hist.empty:
        st.warning("Sem histórico ainda.")
    else:
        prev_view = prev.copy() if not prev.empty else pd.DataFrame()
        if not prev_view.empty:
            prev_view["suspeita"] = prev_view.apply(previsao_suspeita, axis=1)

            suspeitas = prev_view[prev_view["suspeita"]]
            if not suspeitas.empty:
                periodos = ", ".join(suspeitas["periodo"].astype(str).tolist())
                st.markdown(
                    f'<div class="oreilly-callout">'
                    f'<strong>Errata</strong> — as previsões para {periodos} '
                    f'parecem inconsistentes com a série histórica '
                    f'(faixa esperada R${PREVISAO_MIN_RAZOAVEL:.0f}–R${PREVISAO_MAX_RAZOAVEL:.0f}). '
                    f'Provavelmente o modelo precisa ser re-treinado — verifique se há '
                    f'transformação log/diff sem inversão correta.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Constrói série emendada: histórico + ponto de junção + previsão
        ultimo_hist_dt = hist["dt"].max()
        ultimo_hist_v = hist.loc[hist["dt"] == ultimo_hist_dt, "preco_buy_box"].iloc[0]

        fig = go.Figure()

        # Banda de confiança da previsão (atrás de tudo)
        if not prev_view.empty:
            x_band = (
                [ultimo_hist_dt]
                + list(prev_view["dt"])
                + list(prev_view["dt"][::-1])
                + [ultimo_hist_dt]
            )
            y_upper = (
                [ultimo_hist_v]
                + list(prev_view["intervalo_max"])
            )
            y_lower = (
                list(prev_view["intervalo_min"][::-1])
                + [ultimo_hist_v]
            )
            fig.add_trace(go.Scatter(
                x=x_band,
                y=y_upper + y_lower,
                fill="toself",
                fillcolor=RED_SOFT,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                name="Intervalo (95%)",
                showlegend=True,
            ))

        # Histórico — traço sólido escuro
        fig.add_trace(go.Scatter(
            x=hist["dt"],
            y=hist["preco_buy_box"],
            mode="lines+markers",
            name="Histórico",
            line=dict(color=INK, width=2),
            marker=dict(size=5, color=INK),
            hovertemplate="<b>%{x|%b/%Y}</b><br>R$ %{y:.2f}<extra>Histórico</extra>",
        ))

        # Previsão — emenda no último ponto histórico, traço pontilhado vermelho
        if not prev_view.empty:
            x_prev = [ultimo_hist_dt] + list(prev_view["dt"])
            y_prev = [ultimo_hist_v] + list(prev_view["preco_previsto"])
            fig.add_trace(go.Scatter(
                x=x_prev,
                y=y_prev,
                mode="lines+markers",
                name="Previsão",
                line=dict(color=RED, width=2.5, dash="dot"),
                marker=dict(size=7, color=RED, symbol="diamond"),
                hovertemplate="<b>%{x|%b/%Y}</b><br>R$ %{y:.2f}<extra>Previsão</extra>",
            ))

            # Linha vertical separando histórico de previsão
            # Evita bug do Plotly ao usar add_vline com annotation_text em datas Pandas.
            ultimo_hist_dt = pd.to_datetime(ultimo_hist_dt, errors="coerce")

            if pd.notna(ultimo_hist_dt):
                x_hoje = ultimo_hist_dt.to_pydatetime()

                fig.add_shape(
                    type="line",
                    x0=x_hoje,
                    x1=x_hoje,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(color=INK_SOFT, width=1, dash="dash"),
                )

                fig.add_annotation(
                    x=x_hoje,
                    y=1,
                    xref="x",
                    yref="paper",
                    text="hoje",
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(family="Oswald", size=11, color=INK_SOFT),
                )

        fig.update_layout(
            paper_bgcolor=CREAM,
            plot_bgcolor=PAPER,
            font=dict(family="Crimson Pro, Georgia, serif", color=INK, size=13),
            xaxis=dict(
                title=None,
                gridcolor=RULE,
                linecolor=INK,
                ticks="outside",
                tickfont=dict(family="Oswald", size=11),
            ),
            yaxis=dict(
                title=dict(text="Preço (R$)", font=dict(family="Oswald", size=12)),
                gridcolor=RULE,
                linecolor=INK,
                ticks="outside",
                tickfont=dict(family="Oswald", size=11),
                tickprefix="R$ ",
            ),
            hovermode="x unified",
            height=460,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family="Oswald", size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig, width="stretch")

        # Tabela de previsões (compacta) abaixo do gráfico
        if not prev_view.empty:
            st.markdown(
                '<div class="oreilly-eyebrow" style="margin-top:1rem">Tabela — previsões</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                prev_view[[
                    "periodo", "preco_previsto", "intervalo_min",
                    "intervalo_max", "abaixo_threshold", "threshold_usado",
                    "gerado_em",
                ]].rename(columns={
                    "periodo": "Mês",
                    "preco_previsto": "Previsto",
                    "intervalo_min": "Min (95%)",
                    "intervalo_max": "Max (95%)",
                    "abaixo_threshold": "Abaixo do alvo",
                    "threshold_usado": "Alvo",
                    "gerado_em": "Gerado em",
                }),
                width="stretch",
                hide_index=True,
                column_config={
                    "Previsto": st.column_config.NumberColumn(format="R$ %.2f"),
                    "Min (95%)": st.column_config.NumberColumn(format="R$ %.2f"),
                    "Max (95%)": st.column_config.NumberColumn(format="R$ %.2f"),
                    "Alvo": st.column_config.NumberColumn(format="R$ %.2f"),
                },
            )

st.markdown(
    '<div class="oreilly-footer">'
    'Coleta diária via GitHub Actions · Supabase Postgres · '
    'Streamlit Cloud · Modelagem com Holt-Winters em log-preço'
    '</div>',
    unsafe_allow_html=True,
)
