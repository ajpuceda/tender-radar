# Tender Radar

Mistral Workflow que escanea diariamente las compras públicas de IA en Polonia (BZP / e-Zamówienia) y la UE (TED) y publica un digest en español en una biblioteca de documentos del workspace.

## Qué hace

1. **BZP** (`ezamowienia.gov.pl`): busca anuncios de contrato con palabras clave de IA en polaco (rate limit 25 req/min).
2. **TED** (`api.ted.europa.eu`): búsqueda full-text de términos de IA en inglés (toda la UE).
3. **Clasificación**: `mistral-small-latest` con salida estructurada decide si cada anuncio *realmente* requiere IA (filtrado de falsos positivos) y su relevancia (alta/media/baja) para una consultora IA unipersonal en remoto.
4. **Digest**: markdown en español ordenado por relevancia, subido a la biblioteca `Tender Radar` del workspace.

## Workflow

- `apps/worker/src/worker/workflows/tender_radar.py` — workflow `tender_radar` (input: `days_lookback`, `bzp_keywords`, `ted_terms`, `library_name`, `dry_run`).
- `apps/worker/src/worker/tender_digest.py` — parseo/dedupe/digest puro (sin SDK ni red), testeable aisladamente.

## Desarrollo

```bash
bash tools/uv.sh sync --all-packages          # genera uv.lock si falta
bash tools/uv.sh run --all-packages python -m worker.entrypoints.dev   # worker local con hot-reload
bunx nx run worker:semgrep                    # reglas de determinismo/imports/estructura
```

> Nota: `uv.lock` no está commiteado; el build de Mistral Cloud lo genera (todas las dependencias resuelven desde PyPI público). Si tienes `uv` localmente, ejecuta `bash tools/uv.sh lock`, commitea el lock y restaura `--frozen` en `deploy/docker/Dockerfile.worker`.

## Despliegue

Deployment gestionado en Mistral Cloud: build desde `deploy/docker/Dockerfile.worker` (contexto raíz). Ejecutar primero con `dry_run=true`.
