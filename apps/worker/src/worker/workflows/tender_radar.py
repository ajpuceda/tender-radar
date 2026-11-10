"""Tender Radar workflow: scans BZP (Poland) and TED (EU) for AI-related public
tenders, filters false positives with a Mistral model, and publishes a
markdown digest to a workspace document library.

- ``fetch`` activities call the public procurement APIs with httpx (no auth).
- The classification step uses the Mistral plugin ``chat_parse_to_model`` so a
  vendor can decide the real AI share, as raw keyword hits are mostly noise.
- ``publish_digest`` writes the digest into a Mistral document library through
  ``get_mistral_client()``; with ``dry_run=True`` the digest is only returned.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Literal

import mistralai.workflows as workflows
from mistralai.workflows import workflow
from mistralai.workflows.plugins.mistralai import (
    ChatCompletionRequest,
    SystemMessage,
    UserMessage,
    chat_parse_to_model,
)
from pydantic import BaseModel, ConfigDict, Field

from worker.tender_digest import (
    BZP_BASE_URL,
    DEFAULT_BZP_KEYWORDS,
    DEFAULT_TED_TERMS,
    TED_FIELDS,
    TED_SEARCH_URL,
    ClassifiedTender,
    DigestStats,
    TenderItem,
    build_digest,
    dedupe_items,
    parse_bzp_item,
    parse_ted_notice,
)

CLASSIFICATION_MODEL = "mistral-small-latest"
CLASSIFY_CHUNK_SIZE = 20
BZP_PAGE_SIZE = 50
BZP_MAX_PAGES_PER_KEYWORD = 2
TED_PAGE_LIMIT = 100

BZP_RATE_LIMIT = workflows.RateLimit(
    time_window_in_sec=60, max_execution=25, key="tender-radar-bzp"  # API allows 30 req/min
)


class TenderRadarInput(BaseModel):
    """Workflow input; the entrypoint schema is reflected from it."""

    model_config = ConfigDict(title="Entrada del Tender Radar")

    days_lookback: int = Field(default=2, ge=1, le=14, description="Días hacia atrás a escanear.")
    bzp_keywords: list[str] = Field(
        default_factory=lambda: list(DEFAULT_BZP_KEYWORDS),
        description="Palabras clave a buscar en el objeto del contrato (BZP, Polonia).",
    )
    ted_terms: list[str] = Field(
        default_factory=lambda: list(DEFAULT_TED_TERMS),
        description="Términos full-text a buscar en TED (UE).",
    )
    library_name: str = Field(default="Tender Radar", description="Biblioteca de destino.")
    dry_run: bool = Field(default=False, description="No publicar; solo devolver el digest.")


class PublishResult(BaseModel):
    library_id: str
    document_id: str
    document_name: str


class TenderRadarOutput(BaseModel):
    digest_markdown: str
    date_from: str
    date_to: str
    ai_notice_count: int
    published: bool = False
    publish: PublishResult | None = None


# --------------------------------------------------------------------------
# Activities
# --------------------------------------------------------------------------


def _require_ok(status_code: int) -> None:
    if status_code == 200:
        return
    retryable = status_code in {429, 502, 503, 504}
    raise workflows.WorkflowError(
        f"tender_source_http_{status_code}",
        non_retryable=not retryable,
        next_retry_delay=timedelta(seconds=30) if retryable else None,
    )


@workflows.activity(
    name="tender-radar-bzp-search",
    start_to_close_timeout=timedelta(seconds=180),
    retry_policy_max_attempts=3,
    retry_policy_backoff_coefficient=2.0,
    rate_limit=BZP_RATE_LIMIT,
)
async def bzp_search_notices(
    date_from: str, date_to: str, keywords: list[str]
) -> list[dict[str, Any]]:
    """Search the Polish BZP for tender objects matching the AI keywords."""
    import httpx

    collected: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        for keyword in keywords:
            for page in range(1, BZP_MAX_PAGES_PER_KEYWORD + 1):
                response = await client.get(
                    BZP_BASE_URL,
                    params={
                        "Page": page,
                        "PageSize": BZP_PAGE_SIZE,
                        "NoticeType": "ContractNotice",
                        "OrderObject": keyword,
                        "PublicationDateFrom": date_from,
                        "PublicationDateTo": date_to,
                    },
                )
                if response.status_code == 403:
                    # Rate limited by the platform: back off and retry once.
                    raise workflows.WorkflowError(
                        "bzp_rate_limited", next_retry_delay=timedelta(seconds=65)
                    )
                _require_ok(response.status_code)
                try:
                    batch = response.json()
                except ValueError:
                    raise workflows.WorkflowError("bzp_invalid_response", non_retryable=True) from None
                if not isinstance(batch, list):
                    raise workflows.WorkflowError("bzp_invalid_response", non_retryable=True)
                collected.extend(batch)
                if len(batch) < BZP_PAGE_SIZE:
                    break
    return collected


@workflows.activity(
    name="tender-radar-ted-search",
    start_to_close_timeout=timedelta(seconds=120),
    retry_policy_max_attempts=3,
    retry_policy_backoff_coefficient=2.0,
)
async def ted_search_notices(date_from: str, date_to: str, terms: list[str]) -> list[dict[str, Any]]:
    """Search TED (EU) full text for AI-related notices in the window."""
    import httpx

    if not terms:
        return []
    quoted = " OR ".join(f'FT="{term}"' for term in terms)
    # PD is the publication date (YYYYMMDD); lower-bounded by the window start.
    query = f"({quoted}) AND PD>={date_from.replace('-', '')}"
    payload = {
        "query": query,
        "limit": TED_PAGE_LIMIT,
        "fields": list(TED_FIELDS),
    }
    async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
        response = await client.post(TED_SEARCH_URL, json=payload)
        _require_ok(response.status_code)
        try:
            body = response.json()
        except ValueError:
            raise workflows.WorkflowError("ted_invalid_response", non_retryable=True) from None
    notices = body.get("notices")
    if not isinstance(notices, list):
        raise workflows.WorkflowError("ted_invalid_response", non_retryable=True)
    return notices


@workflows.activity(
    name="tender-radar-publish",
    start_to_close_timeout=timedelta(seconds=120),
    retry_policy_max_attempts=1,  # an upload retry would duplicate the digest
)
async def publish_digest(filename: str, markdown: str, library_name: str) -> PublishResult:
    """Upload the digest to a workspace document library, creating it if needed."""
    import io

    from mistralai.workflows.plugins.mistralai import get_mistral_client

    try:  # the File helper moved between SDK minor versions
        from mistralai.client.models import File
    except ImportError:  # pragma: no cover - depends on the installed SDK
        from mistralai.models import File

    client = get_mistral_client()
    library_id: str | None = None
    try:
        listed = client.beta.libraries.list()
        for candidate in getattr(listed, "data", []) or []:
            if str(getattr(candidate, "name", "")).strip().lower() == library_name.strip().lower():
                library_id = candidate.id
                break
    except Exception:
        library_id = None
    if library_id is None:
        created = client.beta.libraries.create(
            name=library_name,
            description="Digests automáticos de tenders de IA (BZP Polonia + TED UE) generados por el workflow tender-radar.",
        )
        library_id = created.id
    upload = client.beta.libraries.documents.upload(
        library_id=library_id,
        file=File(fileName=filename, content=io.BytesIO(markdown.encode("utf-8"))),
    )
    return PublishResult(
        library_id=library_id,
        document_id=str(getattr(upload, "id", "")),
        document_name=str(getattr(upload, "name", filename)),
    )


# --------------------------------------------------------------------------
# Classification (Mistral plugin)
# --------------------------------------------------------------------------


class ClassifiedItem(BaseModel):
    reference: str
    is_ai: bool
    relevance: Literal["alta", "media", "baja"]
    summary_es: str = ""
    reason: str = ""


class ClassifiedBatch(BaseModel):
    items: list[ClassifiedItem]


CLASSIFY_SYSTEM_PROMPT = (
    "Eres un analista de compras públicas especializado en proyectos de IA. "
    "Recibes anuncios de licitación y decides, para cada uno: (1) si realmente requiere "
    "implementar, desarrollar, suministrar o prestar servicios de inteligencia artificial "
    "(chatbots, LLM, visión por computador, automatización con IA, plataformas de IA, "
    "formación específica en IA) — NO marque como IA los anuncios que solo mencionan 'digital', "
    "'inteligente' o IT genérico; (2) la relevancia para una consultora/implantadora de IA "
    "unipersonal con EMBA, en remoto, con la plataforma Mistral, presupuesto pequeño: "
    "'alta' si puede ofertar como subcontratista o directamente (incluidos servicios de "
    "formación en IA), 'media' si necesita socio o es de gran escala, 'baja' si solo compra "
    "hardware/licencias; (3) un resumen de una línea en español del objeto del contrato; "
    "(4) una razón breve en español. Responde SOLO con el JSON pedido."
)


def _to_classify_payload(item: TenderItem) -> dict[str, str]:
    return {
        "reference": item.reference,
        "source": item.source,
        "title": item.title[:600],
        "buyer": item.buyer[:200],
        "country": item.country,
        "cpv": item.cpv[:120],
        "value": item.value[:60],
        "deadline": item.deadline[:40],
    }


async def classify_items(items: list[TenderItem]) -> list[ClassifiedTender]:
    """Classify candidates in chunks with structured output from Mistral."""
    classified: list[ClassifiedTender] = []
    by_reference = {item.reference: item for item in items}
    for start in range(0, len(items), CLASSIFY_CHUNK_SIZE):
        chunk = items[start : start + CLASSIFY_CHUNK_SIZE]
        listing = "\n".join(
            f"{index}. {payload['reference']} | {payload['source']} | {payload['title']}"
            for index, payload in enumerate(
                (_to_classify_payload(item) for item in chunk), start=1
            )
        )
        request = ChatCompletionRequest(
            model=CLASSIFICATION_MODEL,
            messages=[
                SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
                UserMessage(
                    content=(
                        "Clasifica cada anuncio (referencia = campo reference). "
                        "Devuelve un item por anuncio con la referencia exacta.\n\n" + listing
                    )
                ),
            ],
        )
        batch = await chat_parse_to_model(ClassifiedBatch, request)
        for entry in batch.items:
            original = by_reference.get(entry.reference)
            if original is None:
                continue
            classified.append(
                ClassifiedTender(
                    item=original,
                    is_ai=entry.is_ai,
                    relevance=entry.relevance,
                    summary_es=entry.summary_es,
                    reason=entry.reason,
                )
            )
    return classified


# --------------------------------------------------------------------------
# Workflow
# --------------------------------------------------------------------------


@workflows.workflow.define(name="tender_radar")
class TenderRadarWorkflow:
    """Scans BZP + TED for AI tenders and publishes the digest to a library."""

    @workflows.workflow.entrypoint
    async def run(self, data: TenderRadarInput) -> TenderRadarOutput:
        today = workflow.now().date()
        date_to = f"{today:%Y-%m-%d}"
        date_from = f"{today - timedelta(days=data.days_lookback):%Y-%m-%d}"

        bzp_raw, ted_raw = await asyncio.gather(  # noqa: FLY002 — deterministic fan-out over activities
            bzp_search_notices(date_from, date_to, data.bzp_keywords),
            ted_search_notices(date_from, date_to, data.ted_terms),
        )

        candidates: list[TenderItem] = []
        for raw in bzp_raw:
            parsed = parse_bzp_item(raw)
            if parsed is not None:
                candidates.append(parsed)
        bzp_count = len(candidates)
        for raw in ted_raw:
            parsed = parse_ted_notice(raw)
            if parsed is not None:
                candidates.append(parsed)
        ted_count = len(candidates) - bzp_count

        unique = dedupe_items(candidates)
        classified = await classify_items(unique)
        ai_items = [entry for entry in classified if entry.is_ai]
        stats = DigestStats(
            bzp_candidates=bzp_count,
            ted_candidates=ted_count,
            unique_candidates=len(unique),
            ai_notices=len(ai_items),
            high_relevance=sum(1 for entry in ai_items if entry.relevance == "alta"),
        )
        digest = build_digest(classified, date_from, date_to, stats)

        if data.dry_run:
            return TenderRadarOutput(
                digest_markdown=digest,
                date_from=date_from,
                date_to=date_to,
                ai_notice_count=stats.ai_notices,
                published=False,
            )

        timestamp = f"{workflow.now():%Y-%m-%d_%H-%M}"
        filename = f"tender-radar_{timestamp}.md"
        publish = await publish_digest(filename, digest, data.library_name)
        return TenderRadarOutput(
            digest_markdown=digest,
            date_from=date_from,
            date_to=date_to,
            ai_notice_count=stats.ai_notices,
            published=True,
            publish=publish,
        )
