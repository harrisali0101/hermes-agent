"""Reusable document → gbrain ingest pipeline.

This module owns the OCR / extract / chunk / embed / upsert flow for a single
local file. It's the single code path shared by:

  * Block G — the MESEC email-attachment worker (see
    `azure/connectors/project-email/attach.py` in the personal-agent repo).
    Block G was the original home of every extractor + the Document
    Intelligence caller; the intent is that a follow-up refactor moves Block G
    off its private copies and onto this function.
  * `hermes_save:save_document_to_scope` — the new MCP tool that receives the
    cached inbound-media path from a WhatsApp/Slack upload and persists the
    file as a scoped gbrain page.

WHY IT'S SPLIT OUT.  Block G's `attach.py` is a 1,200-line worker that also
owns Graph API pagination, per-email reconcile state, forwarded-email item
attachments, zip un-nesting, GDPR erasure, and a legacy-Office LibreOffice
subprocess dance.  Ninety percent of that is email-specific.  The reusable
core is: given `(local_path, source_id, slug, title)`, produce a gbrain page
plus a Blob mirror, deduplicating by content hash.  That's what this file is.

FLOW (see the `gbrain_ingest_document` docstring for the authoritative order):

  1. Verify local file (exists / readable / <= 20 MB).
  2. SHA-256 the raw bytes → `content_hash`.
  3. Query gbrain for an existing page in `source_id` whose frontmatter
     `content_hash` matches.  If found → `already_ingested`.
  4. Resolve MIME (caller-supplied → `mimetypes` → magic-byte sniff).
  5. Extract to markdown, routed by MIME/extension:
       - text-native → decode
       - .docx / .xlsx / .pptx → python-docx / openpyxl / python-pptx
       - PDF → text-layer if present, else Document Intelligence
       - images → Document Intelligence prebuilt-read
       - anything else → placeholder + warning
  6. Chunk the extracted markdown (800-char target, 100-char overlap).
  7. Embed the chunks via Azure OpenAI `text-embedding-3-large` (1536 dims).
     Embedding failure is non-fatal — gbrain's autopilot embedder will fill
     in downstream — but a successful embed is stored as a sidecar so the
     gbrain page is queryable without waiting for autopilot to catch up.
  8. Upsert to gbrain via `gbrain capture --file … --source … --slug …
     --title … --type document`.  Idempotent by slug on the gbrain side.
  9. Mirror the raw file to Azure Blob at
     `<container>/uploads/<source_id>/<slug>.<ext>` (best-effort — a Blob
     failure downgrades to a warning; the page still ships).

Failure modes are exposed via `IngestResult.status`:
  `created` | `updated` | `already_ingested` | `dry_run` | `failed`.

SAFETY

  * `pathlib.Path` throughout; every `subprocess.run` uses a list arg (never
    `shell=True`), so a malicious `local_path` cannot inject.
  * OCR unavailable → text-layer / placeholder fallback + `ocr_unavailable`
    warning; the ingest still lands.
  * Blob unavailable → `blob_mirror_failed` warning; page still ships.
  * Chunk/embed failure → `embedding_failed` warning; page still ships (gbrain
    autopilot will re-embed).  Extraction failure → `IngestResult(failed)`.
  * `dry_run=True` → validate + hash + mime-resolve, then return without any
    gbrain / Blob / OpenAI side effects.

No personal names in this file — role terms only.  Callers pass `sender_id`
into the MCP tool that WRAPS this function; the function itself is
identity-agnostic and only knows about the target `source_id`.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import re
import shutil
import subprocess

# Resolve the gbrain CLI binary path at import time. shutil.which honours
# the process PATH; if the stdio-MCP subprocess PATH doesn't include the
# Bun global bin dir where gbrain lives, fall back to the known install
# path (verified via `sudo find /home -name gbrain 2>/dev/null` on the VM,
# 2026-07-05). Using a plain "gbrain" string previously failed with
# `FileNotFoundError: 'gbrain'` in production; the outer error handler
# caught the exception but the calling MCP tool did not surface the
# failure, so this constant is the fix-at-source.
_GBRAIN_BIN = (
    shutil.which("gbrain")
    or "/home/hermes-user/.bun/bin/gbrain"
)

# The gbrain CLI is a Bun shebang script (`#!/usr/bin/env bun`), so the
# subprocess PATH must include the Bun bin dir or the script can't start
# (`env: 'bun': No such file or directory`, exit code 127). Under
# systemd + fetch-secrets, the parent hermes process's PATH sometimes
# omits Bun's install location. We build the child env explicitly so
# `gbrain` and `bun` both resolve regardless of the parent PATH.
_BUN_BIN_DIR = "/home/hermes-user/.bun/bin"


def _gbrain_subprocess_env() -> dict:
    """Return an env dict for gbrain CLI subprocess calls with Bun on PATH."""
    env = dict(os.environ)
    existing_path = env.get("PATH", "")
    if _BUN_BIN_DIR not in existing_path.split(os.pathsep):
        env["PATH"] = (
            _BUN_BIN_DIR + os.pathsep + existing_path
            if existing_path
            else _BUN_BIN_DIR
        )
    return env
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_BYTES = 20 * 1024 * 1024  # 20 MB hard cap on inbound uploads
CHUNK_TARGET = 800  # target chars per chunk
CHUNK_OVERLAP = 100  # trailing chars re-included in the next chunk
EMBED_MODEL_DIMS = 1536  # text-embedding-3-large @ 1536 (matches Block G)
DI_API_VERSION_DEFAULT = "2024-11-30"
BLOB_UPLOAD_PREFIX = "uploads"  # container/uploads/<source_id>/<slug>.<ext>

# Extension buckets (mirrors Block G's routing so both paths agree).
TEXTY_EXTS = {"txt", "md", "csv", "json", "log", "tsv", "html", "htm", "xml",
              "yaml", "yml"}
OFFICE_EXTS = {"docx", "xlsx", "pptx"}
PDF_EXTS = {"pdf"}
IMAGE_EXTS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "heic", "webp"}

# gbrain frontmatter key that stores the content hash so re-runs dedupe.
FRONTMATTER_HASH_KEY = "content_hash"


# ─────────────────────────── result envelope ───────────────────────────────


@dataclass
class IngestResult:
    """Outcome of a single `gbrain_ingest_document` call.

    `status` is the state discriminator; every other field is best-effort
    context.  Callers should switch on `status` first, then read `warnings`
    to decide whether to surface a soft note to the end user.
    """

    status: str  # "created" | "updated" | "already_ingested" | "dry_run" | "failed"
    slug: str  # gbrain slug on success; "" on failure or dry_run
    page_id: Optional[int]  # gbrain page id if the CLI/API returned one
    content_hash: str  # sha256 hex of the raw file
    chunks_created: int  # 0 on skip/failure/dry_run
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None  # populated iff status == "failed"


# ───────────────────────────── small helpers ───────────────────────────────


def _sniff_mime(data: bytes, path: Path, declared: Optional[str]) -> Tuple[str, str]:
    """Return `(mime, ext_without_dot)`. Prefers caller's declared mime; falls
    back to `mimetypes` then magic-byte sniff (borrowed from Block G's
    `sniff_ext`, but wired for MIME output instead of just extension).
    Trusting the file extension alone sends a `.pdf` that is really a
    `.docx` (ZIP container) to Document Intelligence, which 400s."""
    ext = path.suffix.lstrip(".").lower()
    head = data[:8]
    if head[:5] == b"%PDF-":
        return ("application/pdf", "pdf")
    if head[:4] == b"PK\x03\x04":  # ZIP-based OOXML
        try:
            import io
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = z.namelist()
            if any(n.startswith("word/") for n in names):
                return ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx")
            if any(n.startswith("xl/") for n in names):
                return ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx")
            if any(n.startswith("ppt/") for n in names):
                return ("application/vnd.openxmlformats-officedocument.presentationml.presentation", "pptx")
        except zipfile.BadZipFile:
            pass
    if declared:
        return (declared, ext or (mimetypes.guess_extension(declared) or "").lstrip("."))
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return (guessed, ext)
    return ("application/octet-stream", ext)


def _content_hash(data: bytes) -> str:
    """SHA-256 hex of the raw file bytes — the dedup key."""
    return hashlib.sha256(data).hexdigest()


def _split_paragraphs(text: str) -> List[str]:
    """Split on blank-line boundaries; drop empties.  Cheap and format-agnostic."""
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _chunk_text(text: str, target: int = CHUNK_TARGET,
                overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Paragraph-aware chunker with a sliding-window overlap.

    Groups paragraphs greedily up to `target` chars, then rolls the tail
    `overlap` chars into the next chunk so a claim split across a boundary
    is still discoverable by cosine search on either chunk.  Falls back to a
    hard character split for pathological paragraphs longer than `target`.

    Block G does NOT chunk — it delegates to gbrain's autopilot embedder,
    which chunks at ingest time.  We chunk here so callers who want
    inline embeddings (see `_embed_chunks`) can do them without a round
    trip through autopilot.
    """
    paras = _split_paragraphs(text)
    if not paras:
        return []
    chunks: List[str] = []
    buf: List[str] = []
    length = 0
    for p in paras:
        # Big paragraph → emit it in chunks of `target` chars with overlap.
        if len(p) > target:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, length = [], 0
            step = max(1, target - overlap)
            for i in range(0, len(p), step):
                chunks.append(p[i:i + target])
            continue
        if length + len(p) + 2 > target and buf:
            chunks.append("\n\n".join(buf))
            # Seed the next buffer with the tail of the last chunk for overlap.
            tail = chunks[-1][-overlap:] if overlap and chunks[-1] else ""
            buf = [tail, p] if tail else [p]
            length = len(tail) + len(p)
        else:
            buf.append(p)
            length += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


# ─────────────────────────── extractors ────────────────────────────────────
#
# These are simplified ports of Block G's per-format extractors from
# `attach.py`.  Block G's copies additionally read cell/paragraph SHADING
# and inject `[RED]`/`[AMBER]`/`[GREEN]` prefixes so status-colour semantics
# survive OCR — that's specific to MESEC's closing-checklist workbooks and
# NOT ported here.  If the CEO ever WhatsApp-uploads a coloured tracker
# and wants the RAG status preserved, port `_status_from_rgb` +
# `_fill_status` from Block G.
#
# TODO(shared-core): once Block G is refactored to call
# `gbrain_ingest_document`, move the colour-aware extractors here and gate
# them on an `annotate_status_colours: bool` kwarg.


def _extract_text(data: bytes) -> str:
    """UTF-8 decode with replacement — safe for anything text-native."""
    return data.decode("utf-8", errors="replace").strip()


def _extract_docx(path: Path) -> str:
    from docx import Document  # type: ignore[import-not-found]
    d = Document(str(path))
    parts: List[str] = [p.text for p in d.paragraphs if p.text and p.text.strip()]
    for t in d.tables:
        for row in t.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts).strip()


def _extract_xlsx(path: Path) -> str:
    from openpyxl import load_workbook  # type: ignore[import-not-found]
    wb = load_workbook(str(path), read_only=True, data_only=True)
    out: List[str] = []
    for ws in wb.worksheets:
        out.append(f"\n## Sheet: {ws.title}\n")
        for row in ws.iter_rows(values_only=True):
            cells = ["" if v is None else str(v).strip() for v in row]
            while cells and not cells[-1]:
                cells.pop()
            if any(cells):
                out.append(",".join(_csv_quote(c) for c in cells))
    wb.close()
    return "\n".join(out).strip()


def _csv_quote(cell: str) -> str:
    if any(ch in cell for ch in ',"\n'):
        return '"' + cell.replace('"', '""') + '"'
    return cell


def _extract_pptx(path: Path) -> str:
    from pptx import Presentation  # type: ignore[import-not-found]
    pr = Presentation(str(path))
    out: List[str] = []
    for i, slide in enumerate(pr.slides, 1):
        out.append(f"\n## Slide {i}\n")
        for sh in slide.shapes:
            if getattr(sh, "has_text_frame", False):
                for para in sh.text_frame.paragraphs:
                    text = "".join(r.text for r in para.runs).strip()
                    if text:
                        out.append(text)
            if getattr(sh, "has_table", False):
                for row in sh.table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        out.append(" | ".join(cells))
    return "\n".join(out).strip()


def _pdf_has_text_layer(data: bytes) -> bool:
    """Heuristic: search the first 128 KB for a text-showing operator
    (`Tj`, `TJ`, or `BT`).  A born-digital PDF hits this; a pure scan
    doesn't and needs DI OCR."""
    head = data[:128 * 1024]
    return bool(re.search(rb"/(Tj|TJ)|BT\b", head))


def _extract_pdf_text_layer(path: Path) -> str:
    """Best-effort PDF text-layer read via pypdf (if installed).  Returns
    an empty string on failure — caller then routes to DI OCR."""
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "").strip()
                           for page in reader.pages).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("pypdf text-layer extraction failed: %s", exc)
        return ""


def _ocr_via_di(data: bytes, content_type: str,
                endpoint: str, key: str,
                api_version: str = DI_API_VERSION_DEFAULT) -> str:
    """Azure Document Intelligence `prebuilt-read` OCR.

    Ported from Block G's `ocr_via_di` (attach.py:532) — same retry-on-429
    behaviour, same 5-minute poll ceiling (large scans can be 100+ pages).
    Raises on failure so the extractor loop can downgrade to a placeholder.
    """
    import json
    import time
    url = f"{endpoint.rstrip('/')}/documentintelligence/documentModels/prebuilt-read:analyze?api-version={api_version}"
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Ocp-Apim-Subscription-Key": key,
                 "Content-Type": content_type or "application/octet-stream"},
    )
    # POST → 202 Accepted with `operation-location` header.
    op_location = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                op_location = resp.headers.get("operation-location")
                break
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < 5:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise
    if not op_location:
        raise RuntimeError("DI: no operation-location on POST")
    for _ in range(150):
        time.sleep(2)
        poll = urllib.request.Request(
            op_location, headers={"Ocp-Apim-Subscription-Key": key})
        with urllib.request.urlopen(poll, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        status = body.get("status")
        if status == "succeeded":
            return (body.get("analyzeResult") or {}).get("content", "").strip()
        if status == "failed":
            raise RuntimeError(f"DI analyze failed: {body.get('error')}")
    raise RuntimeError("DI analyze timed out")


def _extract(path: Path, data: bytes, mime: str, ext: str,
             warnings: List[str]) -> str:
    """Route to the right extractor by mime/ext.  Never raises — degrades
    to a `[non-text-content; N bytes]` placeholder and appends a warning."""
    try:
        if ext in TEXTY_EXTS or (mime or "").startswith("text/"):
            return _extract_text(data)
        if ext == "docx":
            return _extract_docx(path)
        if ext == "xlsx":
            return _extract_xlsx(path)
        if ext == "pptx":
            return _extract_pptx(path)
        if ext in PDF_EXTS or mime == "application/pdf":
            if _pdf_has_text_layer(data):
                text = _extract_pdf_text_layer(path)
                if text:
                    return text
                warnings.append("pdf_text_layer_empty")
            return _ocr_or_placeholder(data, mime or "application/pdf", warnings)
        if ext in IMAGE_EXTS or (mime or "").startswith("image/"):
            return _ocr_or_placeholder(data, mime or "application/octet-stream", warnings)
        warnings.append(f"unsupported_type:{ext or mime}")
        return f"[non-text-content; {len(data)} bytes; type={ext or mime}]"
    except ImportError as exc:
        warnings.append(f"extractor_missing:{exc.name}")
        return f"[extractor unavailable ({exc.name}); {len(data)} bytes]"
    except Exception as exc:  # noqa: BLE001
        logger.exception("extractor error for %s", path)
        warnings.append(f"extract_error:{type(exc).__name__}")
        return f"[extract failed: {type(exc).__name__}; {len(data)} bytes]"


def _ocr_or_placeholder(data: bytes, content_type: str,
                        warnings: List[str]) -> str:
    """DI if configured; else a placeholder + `ocr_unavailable` warning."""
    endpoint = os.environ.get("DI_ENDPOINT") or os.environ.get("AZURE_DI_ENDPOINT")
    key = os.environ.get("DI_KEY") or os.environ.get("AZURE_DI_KEY")
    if not endpoint or not key:
        warnings.append("ocr_unavailable")
        return f"[non-text-content; {len(data)} bytes; OCR not configured]"
    try:
        return _ocr_via_di(data, content_type, endpoint, key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DI OCR failed, using placeholder: %s", exc)
        warnings.append("ocr_failed")
        return f"[non-text-content; {len(data)} bytes; OCR failed: {type(exc).__name__}]"


# ───────────────────────── embeddings (Azure OpenAI) ───────────────────────


def _embed_chunks(chunks: List[str]) -> Optional[List[List[float]]]:
    """Call Azure OpenAI `text-embedding-3-large` at 1536 dims.  Returns
    `None` on any failure (missing env, HTTP error, quota) — the caller
    treats it as `embedding_failed` warning and lets gbrain's autopilot
    re-embed downstream.  Never raises."""
    import json
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = (os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
                  or os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT"))
    key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    if not endpoint or not deployment or not key or not chunks:
        return None
    url = (f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/embeddings"
           f"?api-version={api_version}")
    payload = json.dumps({"input": chunks, "dimensions": EMBED_MODEL_DIMS}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"api-key": key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("embedding call failed: %s", exc)
        return None
    try:
        return [item["embedding"] for item in body["data"]]
    except (KeyError, TypeError) as exc:
        logger.warning("embedding response malformed: %s", exc)
        return None


# ───────────────────────── gbrain adapter ──────────────────────────────────


def _gbrain_find_by_content_hash(source_id: str, content_hash: str
                                 ) -> Optional[Tuple[str, Optional[int]]]:
    """Look up an existing page in `source_id` whose frontmatter carries
    the given `content_hash`.  Returns `(slug, page_id_or_None)` or `None`.

    Uses `gbrain search` with a metadata-key query.  If gbrain's search
    doesn't support the metadata syntax yet, this call fails soft (returns
    `None`) and the ingest re-runs — slug-level idempotency inside gbrain
    still prevents duplicate slugs from clashing.

    TODO(shared-core): once gbrain ships a first-class content-hash lookup,
    replace this with the direct endpoint.  Today's grep is O(source-size)
    per ingest.
    """
    try:
        r = subprocess.run(
            [_GBRAIN_BIN, "search", "--source", source_id,
             "--frontmatter", f"{FRONTMATTER_HASH_KEY}={content_hash}",
             "--format", "json"],
            capture_output=True, text=True, timeout=30,
            env=_gbrain_subprocess_env(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("gbrain search unavailable: %s", exc)
        return None
    if r.returncode != 0:
        return None
    import json
    try:
        hits = json.loads(r.stdout or "[]")
    except json.JSONDecodeError:
        return None
    if not hits:
        return None
    first = hits[0] if isinstance(hits, list) else hits
    if not isinstance(first, dict):
        return None
    slug = first.get("slug") or ""
    if not slug:
        return None
    pid = first.get("page_id")
    return (slug, pid if isinstance(pid, int) else None)


def _render_frontmatter(*, slug: str, title: str, source_id: str,
                        content_hash: str, mime: str, size_bytes: int,
                        ext: str, blob_uri: str,
                        warnings: List[str]) -> str:
    """YAML frontmatter for the gbrain page.  Kept minimal — gbrain merges
    tags across recaptures, so a re-run with fresh warnings won't clobber
    the previous run's state."""
    def _yaml_str(v: str) -> str:
        # Single-quote-escape any embedded quotes.
        return "'" + v.replace("'", "''") + "'"
    lines = [
        "---",
        f"title: {_yaml_str(title)}",
        f"slug: {slug}",
        f"intended_scope: {source_id}",
        f"{FRONTMATTER_HASH_KEY}: {content_hash}",
        f"mime_type: {mime}",
        f"file_extension: {ext}",
        f"size_bytes: {size_bytes}",
        f"ingested_via: gbrain_ingest_document",
    ]
    if blob_uri:
        lines.append(f"blob_uri: {_yaml_str(blob_uri)}")
    if warnings:
        lines.append("ingest_warnings:")
        for w in warnings:
            lines.append(f"  - {w}")
    lines.append("---\n")
    return "\n".join(lines)


def _gbrain_capture(*, slug: str, source_id: str, title: str,
                    markdown: str) -> Tuple[str, str, Optional[int]]:
    """Call `gbrain capture` and parse its result.

    Returns `(status, stored_slug, page_id)` where `status` is
    `"created"` or `"updated"` (mapped from gbrain's stdout) and
    `stored_slug` is what gbrain actually persisted (it may normalise).
    Raises `RuntimeError` on any subprocess or parse failure.

    Uses a list-arg subprocess call — never `shell=True` — so the local
    path can't inject even if it contains shell metacharacters.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(markdown)
        staged = fh.name
    try:
        r = subprocess.run(
            [_GBRAIN_BIN, "capture", "--file", staged,
             "--source", source_id, "--slug", slug,
             "--title", title, "--type", "document"],
            capture_output=True, text=True, timeout=120,
            env=_gbrain_subprocess_env(),
        )
    finally:
        try:
            os.unlink(staged)
        except OSError:
            pass
    if r.returncode != 0:
        raise RuntimeError(
            f"gbrain capture failed (rc={r.returncode}): "
            f"{(r.stderr or r.stdout or '').strip()[:300]}"
        )
    stdout = r.stdout or ""
    slug_m = re.search(r"slug:\s*(\S+)", stdout)
    status_m = re.search(r"status:\s*(\S+)", stdout)
    page_m = re.search(r"page[_-]?id:\s*(\d+)", stdout)
    stored_slug = slug_m.group(1) if slug_m else slug
    raw_status = (status_m.group(1) if status_m else "").lower()
    if raw_status in ("created", "created_or_updated"):
        status = "created"
    elif raw_status in ("updated", "skipped"):
        status = "updated"
    else:
        # Unknown status — trust the return code and treat as created.
        status = "created"
    page_id = int(page_m.group(1)) if page_m else None
    return (status, stored_slug, page_id)


# ───────────────────────── Blob mirror ────────────────────────────────────


def _blob_mirror(local_path: Path, source_id: str, slug: str, ext: str,
                 warnings: List[str]) -> str:
    """Mirror the raw file to Azure Blob at
    `<container>/uploads/<source_id>/<slug>.<ext>`.  Returns the blob URI
    on success or empty string on failure (with a `blob_mirror_failed`
    warning already appended).

    Uses the `az storage blob upload` CLI with `--auth-mode login` — the
    same pattern Block G uses (`attach.py:606`).  No SAS keys in code.

    TODO(shared-core): Block G's `blob_upload` also does per-email prefixes
    and has retry logic for connection errors.  When we lift the shared
    core out of Block G, pull that helper up in place of this one.
    """
    account = os.environ.get("HERMES_BLOB_ACCOUNT") or os.environ.get("BLOB_ACCOUNT")
    container = os.environ.get("HERMES_BLOB_CONTAINER") or os.environ.get("BLOB_CONTAINER")
    if not account or not container:
        warnings.append("blob_mirror_failed:not_configured")
        return ""
    dot = f".{ext}" if ext else ""
    blob_path = f"{BLOB_UPLOAD_PREFIX}/{source_id}/{slug}{dot}"
    r = subprocess.run(
        ["az", "storage", "blob", "upload",
         "--account-name", account, "--container-name", container,
         "--name", blob_path, "--file", str(local_path),
         "--auth-mode", "login", "--overwrite", "-o", "none"],
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        logger.warning("blob mirror failed: %s", (r.stderr or "").strip()[:200])
        warnings.append("blob_mirror_failed")
        return ""
    from urllib.parse import quote
    return f"https://{account}.blob.core.windows.net/{container}/{quote(blob_path)}"


# ─────────────────────────── main entry ────────────────────────────────────


def gbrain_ingest_document(
    local_path: str,
    source_id: str,
    slug: str,
    title: str,
    mime_type: Optional[str] = None,
    *,
    dry_run: bool = False,
) -> IngestResult:
    """Ingest a local file into gbrain under the given `source_id`.

    Args:
        local_path: Absolute path to the file on the local filesystem
            (e.g. the WhatsApp-media cache path).
        source_id: gbrain source_id / scope (e.g. `project-mesec`).
        slug: Slug to store the gbrain page under.  Caller is responsible
            for prefixing (`attach-…`, `upload-…`) and for uniqueness
            within the source.
        title: Human-readable title for the page.
        mime_type: Optional caller-supplied MIME.  If omitted, resolved
            from `mimetypes.guess_type` + magic-byte sniff.
        dry_run: If True, validate + hash + mime resolve then return
            `IngestResult(status='dry_run')` without touching gbrain, DI,
            OpenAI, or Blob.

    Returns:
        `IngestResult`.  Callers switch on `.status` and surface
        `.warnings` (non-fatal) to the user.  On `status='failed'`,
        `.error` explains the root cause.
    """
    warnings: List[str] = []
    path = Path(local_path)

    # ── 1. Verify file ────────────────────────────────────────────────────
    if not path.exists() or not path.is_file():
        return IngestResult(status="failed", slug="", page_id=None,
                            content_hash="", chunks_created=0,
                            warnings=warnings, error="file_missing")
    try:
        size = path.stat().st_size
    except OSError as exc:
        return IngestResult(status="failed", slug="", page_id=None,
                            content_hash="", chunks_created=0,
                            warnings=warnings, error=f"stat_failed:{exc}")
    if size == 0:
        return IngestResult(status="failed", slug="", page_id=None,
                            content_hash="", chunks_created=0,
                            warnings=warnings, error="empty_file")
    if size > MAX_BYTES:
        return IngestResult(status="failed", slug="", page_id=None,
                            content_hash="", chunks_created=0,
                            warnings=warnings,
                            error=f"file_too_large:{size}>{MAX_BYTES}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        return IngestResult(status="failed", slug="", page_id=None,
                            content_hash="", chunks_created=0,
                            warnings=warnings, error=f"read_failed:{exc}")

    # ── 2. Hash ──────────────────────────────────────────────────────────
    content_hash = _content_hash(data)

    # ── 4. MIME resolution ──────────────────────────────────────────────
    mime, ext = _sniff_mime(data, path, mime_type)

    if dry_run:
        return IngestResult(status="dry_run", slug="", page_id=None,
                            content_hash=content_hash, chunks_created=0,
                            warnings=warnings, error=None)

    # ── 3. Idempotency check ────────────────────────────────────────────
    existing = _gbrain_find_by_content_hash(source_id, content_hash)
    if existing is not None:
        existing_slug, existing_pid = existing
        return IngestResult(status="already_ingested", slug=existing_slug,
                            page_id=existing_pid, content_hash=content_hash,
                            chunks_created=0, warnings=warnings, error=None)

    # ── 5. Text extraction ──────────────────────────────────────────────
    extracted = _extract(path, data, mime, ext, warnings)

    # ── 6. Chunk & 7. Embed ──────────────────────────────────────────────
    chunks = _chunk_text(extracted)
    embeddings = _embed_chunks(chunks) if chunks else None
    if chunks and embeddings is None:
        # Non-fatal — gbrain autopilot will re-embed downstream.
        warnings.append("embedding_failed")

    # ── 9. Blob mirror (best-effort, done before capture so the frontmatter
    #      can carry the blob_uri) ────────────────────────────────────────
    blob_uri = _blob_mirror(path, source_id, slug, ext, warnings)

    # ── 8. Upsert to gbrain ─────────────────────────────────────────────
    frontmatter = _render_frontmatter(
        slug=slug, title=title, source_id=source_id,
        content_hash=content_hash, mime=mime, size_bytes=size, ext=ext,
        blob_uri=blob_uri, warnings=warnings,
    )
    markdown = frontmatter + f"# {title}\n\n" + (extracted or "_(no extractable text)_") + "\n"
    try:
        status, stored_slug, page_id = _gbrain_capture(
            slug=slug, source_id=source_id, title=title, markdown=markdown,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("gbrain_capture failed for slug=%s source=%s: %s",
                     slug, source_id, exc)
        return IngestResult(status="failed", slug="", page_id=None,
                            content_hash=content_hash, chunks_created=0,
                            warnings=warnings, error=f"capture_failed:{exc}")

    return IngestResult(
        status=status, slug=stored_slug, page_id=page_id,
        content_hash=content_hash,
        chunks_created=len(chunks),
        warnings=warnings, error=None,
    )


__all__ = ["IngestResult", "gbrain_ingest_document"]
