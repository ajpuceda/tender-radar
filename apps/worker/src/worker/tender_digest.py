"""Pure parsing, dedupe and digest-building logic for the Tender Radar.

This module is intentionally free of SDK and network imports so its logic can
be unit-tested and linted in isolation. The workflow module in
``worker.workflows.tender_radar`` calls these helpers from deterministic
workflow code; the dataclass boundaries convert to Pydantic models at the
activity edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BZP_BASE_URL = "https://ezamowienia.gov.pl/mo-board/api/v1/notice"
TED_SEARCH_URL = "https://api.ted.europa.eu/v3/notices/search"
TED_NOTICE_URL = "https://ted.europa.eu/en/notice/-/detail/{publication_number}"

DEFAULT_BZP_KEYWORDS: tuple[str, ...] = (
    "sztuczna inteligencja",
    "sztucznej inteligencji",
    "chatbot",
    "voicebot",
    "uczenie maszynowe",
    "wdrożenie AI",
)

DEFAULT_TED_TERMS: tuple[str, ...] = (
    "artificial intelligence",
    "machine learning",
    "chatbot",
    "large language model",
    "AI system",
)

TED_FIELDS: tuple[str, ...] = (
    "notice-title",
    "publication-date",
    "organisation-name-buyer",
    "buyer-country",
    "deadline",
    "estimated-value-proc",
    "estimated-value-cur-proc",
    "notice-type",
)


@dataclass(slots=True)
class TenderItem:
    """A single tender announcement, normalised across BZP and TED."""

    source: str  # "BZP" | "TED"
    reference: str  # BZP notice number or TED publication number
    title: str
    buyer: str = ""
    country: str = ""
    city: str = ""
    published: str = ""  # ISO-ish date string as returned by the source
    deadline: str = ""  # ISO-ish datetime string, empty when unknown
    cpv: str = ""
    value: str = ""  # estimated value with currency, empty when unknown
    url: str = ""


@dataclass(slots=True)
class DigestStats:
    """Aggregated counters reported alongside the digest."""

    bzp_candidates: int = 0
    ted_candidates: int = 0
    unique_candidates: int = 0
    ai_notices: int = 0
    high_relevance: int = 0


def _first_lang_value(value: Any) -> str:
    """Pick a readable string out of TED's language maps.

    TED returns ``{"eng": "...", "deu": ...}``-shaped maps for titles and
    buyer names. Prefer English, then fall back to the first available
    language, then to the raw value when it is already a plain string.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("eng", "en", "MUL"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        for candidate in value.values():
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        if value.values():
            return str(next(iter(value.values())))[:300].strip()
    if isinstance(value, list):
        for candidate in value:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if value is None:
        return ""
    return str(value).strip()


def parse_bzp_item(item: dict[str, Any]) -> TenderItem | None:
    """Normalise one BZP ``/mo-board/api/v1/notice`` item."""
    if not isinstance(item, dict):
        return None
    reference = str(item.get("noticeNumber") or "").strip()
    title = str(item.get("orderObject") or "").strip()
    if not reference or not title:
        return None
    return TenderItem(
        source="BZP",
        reference=reference,
        title=title,
        buyer=str(item.get("organizationName") or "").strip(),
        country=str(item.get("organizationCountry") or "PL").strip(),
        city=str(item.get("organizationCity") or "").strip(),
        published=str(item.get("publicationDate") or "")[:10],
        deadline=str(item.get("submittingOffersDate") or ""),
        cpv=str(item.get("cpvCode") or "").strip(),
    )


def parse_ted_notice(notice: dict[str, Any]) -> TenderItem | None:
    """Normalise one TED Search API v3 notice."""
    if not isinstance(notice, dict):
        return None
    reference = str(notice.get("publication-number") or "").strip()
    if not reference:
        return None
    countries = notice.get("buyer-country")
    country = ""
    if isinstance(countries, list) and countries:
        country = str(countries[0])
    elif isinstance(countries, str):
        country = countries
    value = ""
    amount = notice.get("estimated-value-proc")
    currency = notice.get("estimated-value-cur-proc")
    if amount not in (None, "") and amount is not None:
        value = f"{amount} {currency or ''}".strip()
    return TenderItem(
        source="TED",
        reference=reference,
        title=_first_lang_value(notice.get("notice-title")),
        buyer=_first_lang_value(notice.get("organisation-name-buyer")),
        country=country,
        published=str(notice.get("publication-date") or "")[:10],
        deadline=_first_lang_value(notice.get("deadline")),
        url=TED_NOTICE_URL.format(publication_number=reference),
        value=value,
    )


def dedupe_items(items: list[TenderItem]) -> list[TenderItem]:
    """Drop duplicates by (source, reference), keeping the first occurrence."""
    seen: set[tuple[str, str]] = set()
    unique: list[TenderItem] = []
    for item in items:
        key = (item.source, item.reference)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


RELEVANCE_ORDER = {"alta": 0, "media": 1, "baja": 2}


@dataclass(slots=True)
class ClassifiedTender:
    """A tender plus the model's classification for the digest."""

    item: TenderItem
    is_ai: bool
    relevance: str  # "alta" | "media" | "baja"
    summary_es: str
    reason: str = ""


def sort_classified(items: list[ClassifiedTender]) -> list[ClassifiedTender]:
    """Sort for the digest: relevance, then earliest deadline, then reference."""
    def sort_key(entry: ClassifiedTender) -> tuple[int, str, str]:
        return (
            RELEVANCE_ORDER.get(entry.relevance, 3),
            entry.item.deadline or "9999",
            entry.item.reference,
        )

    return sorted(items, key=sort_key)


def _format_item_line(entry: ClassifiedTender) -> list[str]:
    item = entry.item
    lines: list[str] = []
    head = f"### {entry.relevance.upper()} · {item.source} · {item.reference}"
    lines.append(head)
    lines.append(f"**{item.title[:300]}**" if item.title else "**(sin título)**")
    meta: list[str] = []
    if item.buyer:
        meta.append(f"Comprador: {item.buyer}")
    if item.city or item.country:
        meta.append(f"Lugar: {item.city + ' (' + item.country + ')' if item.city else item.country}")
    if item.deadline:
        meta.append(f"Deadline: {item.deadline}")
    if item.value:
        meta.append(f"Valor estimado: {item.value}")
    if item.cpv:
        meta.append(f"CPV: {item.cpv}")
    if meta:
        lines.append(" · ".join(meta))
    if entry.summary_es:
        lines.append(f"_{entry.summary_es}_")
    if item.url:
        lines.append(f"[Anuncio]({item.url})")
    return lines


def build_digest(
    classified: list[ClassifiedTender],
    date_from: str,
    date_to: str,
    stats: DigestStats,
) -> str:
    """Build the markdown digest published to the document library."""
    ai_items = [entry for entry in classified if entry.is_ai]
    ordered = sort_classified(ai_items)
    lines: list[str] = []
    lines.append(f"# 📡 Tender Radar — IA en compras públicas")
    lines.append("")
    lines.append(
        f"Ventana: **{date_from} → {date_to}** · "
        f"{stats.bzp_candidates} candidatos BZP (Polonia) · "
        f"{stats.ted_candidates} candidatos TED (UE) · "
        f"{stats.unique_candidates} únicos · "
        f"**{stats.ai_notices} avisos de IA reales** "
        f"({stats.high_relevance} de relevancia alta)"
    )
    lines.append("")
    if not ordered:
        lines.append("## Sin avisos de IA relevantes en esta ventana")
        lines.append("")
        lines.append(
            "Ningún anuncio publicado en el periodo pasa el filtro de IA real. "
            "La próxima ejecución vuelve a escanear automáticamente."
        )
        return "\n".join(lines) + "\n"

    current_relevance = ""
    for entry in ordered:
        if entry.relevance != current_relevance:
            current_relevance = entry.relevance
            label = {
                "alta": "⭐ Relevancia ALTA — oportunidades directas",
                "media": "🔵 Relevancia MEDIA — seguimiento",
                "baja": "⚪ Relevancia BAJA — contexto",
            }.get(current_relevance, current_relevance)
            lines.append(f"## {label}")
            lines.append("")
        lines.extend(_format_item_line(entry))
        lines.append("")
    lines.append("---")
    lines.append(
        "_Fuentes: BZP (e-Zamówienia, Polonia) y TED (UE). "
        "Clasificación automática con IA — verifica siempre el anuncio original antes de ofertar._"
    )
    return "\n".join(lines) + "\n"
