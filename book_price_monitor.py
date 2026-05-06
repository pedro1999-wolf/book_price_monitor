import os
import re
import time
import requests

from dataclasses import dataclass, asdict
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Optional, Dict, List

from dotenv import load_dotenv
from supabase import create_client, Client


load_dotenv()


SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

TIMEZONE = ZoneInfo("America/Sao_Paulo")

PLATFORM_TABLE = "book_platform_price_snapshots"
DAILY_BEST_TABLE = "book_daily_best_snapshots"

REQUEST_SLEEP_SECONDS = 1.5
CURRENCY = "BRL"

# Mantenha True para evitar capturar Kindle/e-book como "menor preço".
EXCLUDE_DIGITAL_PRODUCTS = True


BOOKS = [
    {
        "book_key": "data_science_do_zero_2ed",
        "title": "Data science do zero - 2º Edição: noções fundamentais com Python",
        "query": '"Data Science do Zero" "Noções Fundamentais com Python" "2" livro',
        "required_terms": ["data", "science", "zero", "python"],
        "preferred_terms": ["2", "segunda", "edicao", "edicao", "nocoes", "fundamentais"],
    },
    {
        "book_key": "maos_a_obra_ml_sklearn_keras_tensorflow",
        "title": "Mãos à obra: aprendizado de máquina com Scikit-Learn, Keras & TensorFlow: conceitos, ferramentas e técnicas para a construção de sistemas inteligentes",
        "query": '"Mãos à obra" "Scikit-Learn" "Keras" "TensorFlow" livro',
        "required_terms": ["maos", "obra", "scikit", "keras", "tensorflow"],
        "preferred_terms": ["aprendizado", "maquina", "conceitos", "ferramentas", "sistemas", "inteligentes"],
    },
]


PLATFORMS = {
    "amazon": {
        "display_name": "Amazon",
        "type": "amazon",
        "amazon_domain": "amazon.com.br",
    },
    "mercado_livre": {
        "display_name": "Mercado Livre",
        "type": "google_shopping",
        "query_hint": "mercado livre",
        "source_aliases": ["mercado livre", "mercadolivre"],
        "domains": ["mercadolivre.com.br"],
    },
    "shopee": {
        "display_name": "Shopee",
        "type": "google_shopping",
        "query_hint": "shopee",
        "source_aliases": ["shopee"],
        "domains": ["shopee.com.br"],
    },
    "magalu": {
        "display_name": "Magalu",
        "type": "google_shopping",
        "query_hint": "magalu magazine luiza",
        "source_aliases": ["magalu", "magazine luiza"],
        "domains": ["magazineluiza.com.br", "magalu.com"],
    },
}


@dataclass
class BookCandidate:
    book_key: str
    book_title: str
    platform: str
    result_title: Optional[str]
    price: Optional[float]
    currency: str
    source: Optional[str]
    product_url: Optional[str]
    raw_result: Dict[str, Any]


def validate_env() -> None:
    missing = []

    if not SERPAPI_KEY:
        missing.append("SERPAPI_KEY")

    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")

    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")

    if missing:
        raise RuntimeError(f"Variáveis ausentes no .env: {', '.join(missing)}")


def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def current_run_date() -> str:
    now = datetime.now(TIMEZONE)
    return now.date().isoformat()


def serpapi_request(params: Dict[str, Any]) -> Dict[str, Any]:
    url = "https://serpapi.com/search"

    final_params = {
        **params,
        "api_key": SERPAPI_KEY,
    }

    response = requests.get(url, params=final_params, timeout=60)
    response.raise_for_status()

    return response.json()


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""

    text = text.lower()

    replacements = {
        "á": "a",
        "à": "a",
        "â": "a",
        "ã": "a",
        "é": "e",
        "ê": "e",
        "í": "i",
        "ó": "o",
        "ô": "o",
        "õ": "o",
        "ú": "u",
        "ç": "c",
    }

    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_price(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, dict):
        for key in ["value", "raw", "price"]:
            if key in value:
                return normalize_price(value[key])

    if isinstance(value, list):
        for item in value:
            price = normalize_price(item)
            if price is not None:
                return price

    if isinstance(value, str):
        cleaned = value.strip()

        cleaned = cleaned.replace("R$", "")
        cleaned = cleaned.replace("$", "")
        cleaned = cleaned.replace("BRL", "")
        cleaned = cleaned.replace("US", "")
        cleaned = cleaned.strip()

        # Formato brasileiro: 1.234,56
        if "," in cleaned:
            cleaned = cleaned.replace(".", "").replace(",", ".")

        cleaned = re.sub(r"[^\d.]", "", cleaned)

        if cleaned:
            try:
                return float(cleaned)
            except ValueError:
                return None

    return None


def is_digital_product(title: Optional[str]) -> bool:
    normalized = normalize_text(title)

    digital_terms = [
        "kindle",
        "ebook",
        "e book",
        "digital",
        "pdf",
        "audiobook",
        "audio book",
    ]

    return any(term in normalized for term in digital_terms)


def result_matches_book(item_title: Optional[str], book: Dict[str, Any]) -> bool:
    normalized_title = normalize_text(item_title)

    if not normalized_title:
        return False

    if EXCLUDE_DIGITAL_PRODUCTS and is_digital_product(item_title):
        return False

    required_terms = [normalize_text(term) for term in book["required_terms"]]
    preferred_terms = [normalize_text(term) for term in book["preferred_terms"]]

    required_ok = all(term in normalized_title for term in required_terms)
    preferred_hits = sum(1 for term in preferred_terms if term in normalized_title)

    return required_ok and preferred_hits >= 1


def source_matches_platform(
    source: Optional[str],
    url: Optional[str],
    platform_config: Dict[str, Any],
) -> bool:
    source_text = normalize_text(source)
    url_text = (url or "").lower()

    aliases = platform_config.get("source_aliases", [])
    domains = platform_config.get("domains", [])

    if any(normalize_text(alias) in source_text for alias in aliases):
        return True

    if any(domain.lower() in url_text for domain in domains):
        return True

    return False


def extract_google_shopping_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []

    if isinstance(data.get("shopping_results"), list):
        items.extend(data["shopping_results"])

    categorized = data.get("categorized_shopping_results")

    if isinstance(categorized, list):
        for category in categorized:
            category_results = category.get("shopping_results")
            if isinstance(category_results, list):
                items.extend(category_results)

    return items


def extract_amazon_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []

    for key in ["organic_results", "featured_products", "sponsored_results"]:
        if isinstance(data.get(key), list):
            items.extend(data[key])

    return items


def normalize_google_shopping_item(
    item: Dict[str, Any],
    book: Dict[str, Any],
    platform_name: str,
) -> Optional[BookCandidate]:
    result_title = item.get("title")

    if not result_matches_book(result_title, book):
        return None

    price = item.get("extracted_price")

    if price is None:
        price = normalize_price(item.get("price"))

    price = normalize_price(price)

    if price is None:
        return None

    return BookCandidate(
        book_key=book["book_key"],
        book_title=book["title"],
        platform=platform_name,
        result_title=result_title,
        price=price,
        currency=CURRENCY,
        source=item.get("source"),
        product_url=item.get("link") or item.get("product_link"),
        raw_result=item,
    )


def normalize_amazon_item(
    item: Dict[str, Any],
    book: Dict[str, Any],
) -> Optional[BookCandidate]:
    result_title = item.get("title")

    if not result_matches_book(result_title, book):
        return None

    price = item.get("extracted_price")

    if price is None:
        price = normalize_price(item.get("price"))

    if price is None:
        price = normalize_price(item.get("prices"))

    price = normalize_price(price)

    if price is None:
        return None

    return BookCandidate(
        book_key=book["book_key"],
        book_title=book["title"],
        platform="Amazon",
        result_title=result_title,
        price=price,
        currency=CURRENCY,
        source="Amazon",
        product_url=item.get("link") or item.get("product_link"),
        raw_result=item,
    )


def search_amazon_candidates(book: Dict[str, Any], platform_config: Dict[str, Any]) -> List[BookCandidate]:
    params = {
        "engine": "amazon",
        "amazon_domain": platform_config["amazon_domain"],
        "k": book["query"],
        "language": "pt_BR",
    }

    data = serpapi_request(params)
    items = extract_amazon_items(data)

    candidates = []

    for item in items:
        candidate = normalize_amazon_item(item, book)

        if candidate:
            candidates.append(candidate)

    return candidates


def search_google_shopping_platform_candidates(
    book: Dict[str, Any],
    platform_config: Dict[str, Any],
) -> List[BookCandidate]:
    platform_name = platform_config["display_name"]

    query = f'{book["query"]} {platform_config["query_hint"]}'

    params = {
        "engine": "google_shopping",
        "q": query,
        "google_domain": "google.com.br",
        "gl": "br",
        "hl": "pt-BR",
        "location": "Brazil",
    }

    data = serpapi_request(params)
    items = extract_google_shopping_items(data)

    candidates = []

    for item in items:
        product_url = item.get("link") or item.get("product_link")
        source = item.get("source")

        if not source_matches_platform(source, product_url, platform_config):
            continue

        candidate = normalize_google_shopping_item(item, book, platform_name)

        if candidate:
            candidates.append(candidate)

    return candidates


def search_platform_candidates(
    book: Dict[str, Any],
    platform_config: Dict[str, Any],
) -> List[BookCandidate]:
    if platform_config["type"] == "amazon":
        return search_amazon_candidates(book, platform_config)

    return search_google_shopping_platform_candidates(book, platform_config)


def choose_lowest_price(candidates: List[BookCandidate]) -> Optional[BookCandidate]:
    if not candidates:
        return None

    return min(candidates, key=lambda candidate: candidate.price)


def save_platform_snapshot(
    supabase: Client,
    book: Dict[str, Any],
    platform_name: str,
    best_candidate: Optional[BookCandidate],
    all_candidates: List[BookCandidate],
    error_message: Optional[str] = None,
) -> None:
    run_date = current_run_date()

    if best_candidate:
        row = {
            "run_date": run_date,
            "book_key": book["book_key"],
            "book_title": book["title"],
            "platform": platform_name,
            "found_match": True,
            "best_title": best_candidate.result_title,
            "best_price": best_candidate.price,
            "currency": best_candidate.currency,
            "source": best_candidate.source,
            "product_url": best_candidate.product_url,
            "all_candidates": [asdict(candidate) for candidate in all_candidates],
            "error_message": error_message,
        }
    else:
        row = {
            "run_date": run_date,
            "book_key": book["book_key"],
            "book_title": book["title"],
            "platform": platform_name,
            "found_match": False,
            "best_title": None,
            "best_price": None,
            "currency": CURRENCY,
            "source": None,
            "product_url": None,
            "all_candidates": [asdict(candidate) for candidate in all_candidates],
            "error_message": error_message or "Nenhum preço válido encontrado nesta plataforma.",
        }

    supabase.table(PLATFORM_TABLE).upsert(
        row,
        on_conflict="run_date,book_key,platform",
    ).execute()


def save_daily_best_snapshot(
    supabase: Client,
    book: Dict[str, Any],
    platform_results: List[Dict[str, Any]],
) -> None:
    run_date = current_run_date()

    valid_results = [
        result for result in platform_results
        if result.get("found_match") is True and result.get("best_price") is not None
    ]

    if valid_results:
        best_result = min(valid_results, key=lambda result: result["best_price"])

        row = {
            "run_date": run_date,
            "book_key": book["book_key"],
            "book_title": book["title"],
            "found_match": True,
            "best_platform": best_result["platform"],
            "best_title": best_result["best_title"],
            "best_price": best_result["best_price"],
            "currency": best_result["currency"],
            "source": best_result["source"],
            "product_url": best_result["product_url"],
            "platform_results": platform_results,
            "error_message": None,
        }
    else:
        row = {
            "run_date": run_date,
            "book_key": book["book_key"],
            "book_title": book["title"],
            "found_match": False,
            "best_platform": None,
            "best_title": None,
            "best_price": None,
            "currency": CURRENCY,
            "source": None,
            "product_url": None,
            "platform_results": platform_results,
            "error_message": "Nenhum preço válido encontrado em nenhuma plataforma.",
        }

    supabase.table(DAILY_BEST_TABLE).upsert(
        row,
        on_conflict="run_date,book_key",
    ).execute()


def monitor_book_prices() -> None:
    validate_env()
    supabase = get_supabase_client()

    for book in BOOKS:
        print(f"\nLivro: {book['title']}")

        platform_results = []

        for platform_key, platform_config in PLATFORMS.items():
            platform_name = platform_config["display_name"]

            print(f"  Buscando em: {platform_name}")

            try:
                candidates = search_platform_candidates(book, platform_config)
                best_candidate = choose_lowest_price(candidates)

                save_platform_snapshot(
                    supabase=supabase,
                    book=book,
                    platform_name=platform_name,
                    best_candidate=best_candidate,
                    all_candidates=candidates,
                )

                if best_candidate:
                    result = {
                        "platform": platform_name,
                        "found_match": True,
                        "best_title": best_candidate.result_title,
                        "best_price": best_candidate.price,
                        "currency": best_candidate.currency,
                        "source": best_candidate.source,
                        "product_url": best_candidate.product_url,
                    }

                    print(
                        f"    Menor nesta plataforma: R$ {best_candidate.price:.2f} | "
                        f"{best_candidate.result_title}"
                    )
                else:
                    result = {
                        "platform": platform_name,
                        "found_match": False,
                        "best_title": None,
                        "best_price": None,
                        "currency": CURRENCY,
                        "source": None,
                        "product_url": None,
                        "error_message": "Nenhum candidato válido encontrado.",
                    }

                    print("    Nenhum candidato válido encontrado.")

                platform_results.append(result)

            except Exception as error:
                error_message = str(error)

                save_platform_snapshot(
                    supabase=supabase,
                    book=book,
                    platform_name=platform_name,
                    best_candidate=None,
                    all_candidates=[],
                    error_message=error_message,
                )

                platform_results.append(
                    {
                        "platform": platform_name,
                        "found_match": False,
                        "best_title": None,
                        "best_price": None,
                        "currency": CURRENCY,
                        "source": None,
                        "product_url": None,
                        "error_message": error_message,
                    }
                )

                print(f"    Erro: {error_message}")

            time.sleep(REQUEST_SLEEP_SECONDS)

        save_daily_best_snapshot(
            supabase=supabase,
            book=book,
            platform_results=platform_results,
        )


if __name__ == "__main__":
    monitor_book_prices()
