"""
Fechamento mensal do histórico Amazon do livro Data Science do Zero.

Lógica:
1. Determina o mês a fechar:
   - Sem argumento: fecha o MÊS ANTERIOR ao corrente.
   - --periodo YYYY-MM: força um período específico.
2. Calcula a média de best_price da plataforma 'Amazon' em
   book_platform_price_snapshots para o livro data_science_do_zero_2ed
   no período (apenas linhas com found_match=true e best_price not null).
3. Faz UPSERT em historico_precos_amazon (chave única: periodo).
4. Dispara o re-treino do modelo (retrain_forecast.retrain) se existir.

Comportamento quando NÃO há dados no período:
- Por padrão: loga warning e sai com 0 (sucesso).
  → Útil pro cron, que não deve alarmar quando o scraper ainda não rodava
    naquele mês (caso típico do primeiro fechamento depois do go-live).
- Com --strict: sai 2 (falha).
  → Use em disparos manuais onde você quer ser avisado.

Uso:
    python monthly_close.py                              # mês anterior, soft
    python monthly_close.py --periodo 2026-04            # força período
    python monthly_close.py --periodo 2026-04 --strict   # falha se sem dados
    python monthly_close.py --skip-retrain               # pula re-treino
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
    res = (
        sb.table(TABELA_HISTORICO)
        .upsert(payload, on_conflict="periodo")
        .execute()
    )
    return res.data


def trigger_retrain() -> None:
    """Chama o re-treino do modelo, se o módulo existir."""
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Falha com exit 2 se não houver dados Amazon no período. "
             "Sem essa flag, loga warning e sai 0 (cron-friendly).",
    )
    args = parser.parse_args()

    periodo = args.periodo or previous_month(date.today())
    livro = args.livro

    log.info("Fechando período %s para livro %s", periodo, livro)
    sb = get_client()

    media = fetch_amazon_avg(sb, livro, periodo)
    if media is None:
        msg = (
            f"Sem dados Amazon para {livro} em {periodo} — nada a inserir. "
            "Verifique se o scraper rodou no período."
        )
        if args.strict:
            log.error(msg)
            return 2
        log.warning(msg + " (modo soft — saindo 0)")
        # Mesmo sem dados novos, ainda dispara retrain pra atualizar previsões
        # com base no histórico atual (caso o usuário tenha editado o histórico
        # manualmente, ou se o modelo simplesmente foi atualizado).
        if not args.skip_retrain:
            trigger_retrain()
        return 0

    log.info("Média Amazon de %s em %s: R$ %.2f", livro, periodo, media)
    out = upsert_historico(sb, periodo, media)
    log.info("Upsert em %s OK (%d linha(s)).", TABELA_HISTORICO, len(out or []))

    if not args.skip_retrain:
        trigger_retrain()

    return 0


if __name__ == "__main__":
    sys.exit(main())
