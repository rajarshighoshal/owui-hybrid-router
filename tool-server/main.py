"""Tool server for OpenWebUI: file export + web fetch + citation lookup.

OpenAPI-discoverable so OpenWebUI auto-registers each endpoint as a tool
the model can invoke. Stateless: each request is self-contained. Listens
on the internal docker network only — no auth.
"""
from __future__ import annotations

import base64
import csv
import ipaddress
import io
import logging
import os
import re
import socket
import tempfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urljoin, urlparse

import httpx
import pypandoc
import trafilatura
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger("tool-server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="OWUI Tool Server",
    description=(
        "File export (docx/pdf/md/csv), readable web extraction, and DOI→APA "
        "citation lookup. Used by OpenWebUI models via tool calls."
    ),
    version="0.3.0",
)

OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "http://open-webui:8080").rstrip("/")
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY", "")
OPENWEBUI_ATTACH_EXPORTS = os.getenv("OPENWEBUI_ATTACH_EXPORTS", "true").lower() not in {
    "0",
    "false",
    "no",
}


# --- Request models -------------------------------------------------------

class ExportRequest(BaseModel):
    markdown: str = Field(
        ...,
        description=(
            "The markdown content to convert. May include headings, lists, "
            "in-text citations like (Author, 2024), code blocks, tables, links. "
            "Pass the complete final draft/content, not an outline or summary of it."
        ),
    )
    filename: Optional[str] = Field(
        None,
        description=(
            "Output filename WITHOUT extension. Defaults to 'document'. "
            "Non-alphanumeric chars (except - and _) are stripped for safety."
        ),
    )
    title: Optional[str] = Field(
        None,
        description=(
            "Document title for Word/PDF metadata. Shows in the Properties "
            "pane and on the first page if the template includes {{title}}."
        ),
    )


class CsvExportRequest(BaseModel):
    rows: list[dict] = Field(
        ...,
        description=(
            "List of row dicts. The keys of the first row become the column "
            "headers; all rows should share the same keys."
        ),
    )
    filename: Optional[str] = Field(
        None,
        description="Output filename WITHOUT extension. Defaults to 'data'.",
    )


class FetchUrlRequest(BaseModel):
    url: str = Field(
        ...,
        description="The full URL to fetch. Must include http:// or https://.",
    )
    max_chars: int = Field(
        8000,
        ge=500,
        le=50000,
        description=(
            "Maximum characters to return. Pages longer than this are truncated "
            "with a [...truncated] marker. Default 8000."
        ),
    )


class CitationRequest(BaseModel):
    doi: str = Field(
        ...,
        description=(
            "Digital Object Identifier (e.g. '10.1037/0033-2909.131.6.803'). "
            "Leading 'doi:' or 'https://doi.org/' is stripped automatically."
        ),
    )


# --- Helpers --------------------------------------------------------------

def _safe_filename(name: Optional[str], default: str) -> str:
    raw = (name or default).strip() or default
    safe = re.sub(r"[^A-Za-z0-9_\- ]", "", raw).strip().replace(" ", "_")
    return safe or default


def _convert_pandoc(markdown: str, fmt: str, extra_args: list[str]) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        pypandoc.convert_text(
            markdown,
            to=fmt,
            format="markdown",
            outputfile=str(out_path),
            extra_args=extra_args,
        )
        return out_path.read_bytes()
    finally:
        out_path.unlink(missing_ok=True)


def _auth_header_from_request(request: Optional[Request]) -> dict[str, str]:
    inbound_auth = request.headers.get("authorization") if request else ""
    if inbound_auth:
        return {"Authorization": inbound_auth}
    if OPENWEBUI_API_KEY:
        return {"Authorization": f"Bearer {OPENWEBUI_API_KEY}"}
    return {}


def _attach_file_to_openwebui(
    request: Optional[Request],
    data: bytes,
    media_type: str,
    filename: str,
) -> dict[str, Any]:
    if not OPENWEBUI_ATTACH_EXPORTS:
        return {"attached_to_chat": False, "attach_reason": "disabled"}

    chat_id = request.headers.get("x-open-webui-chat-id") if request else ""
    message_id = request.headers.get("x-open-webui-message-id") if request else ""
    headers = _auth_header_from_request(request)
    if not chat_id or not message_id:
        return {"attached_to_chat": False, "attach_reason": "missing chat/message headers"}
    if not headers:
        return {"attached_to_chat": False, "attach_reason": "missing OpenWebUI auth"}

    try:
        with httpx.Client(timeout=20.0) as client:
            upload = client.post(
                f"{OPENWEBUI_BASE_URL}/api/v1/files/",
                headers={**headers, "Accept": "application/json"},
                files={"file": (filename, data, media_type)},
                data={"process": "false"},
            )
            upload.raise_for_status()
            uploaded = upload.json()
            file_id = uploaded.get("id") or uploaded.get("file", {}).get("id")
            if not file_id:
                return {
                    "attached_to_chat": False,
                    "attach_reason": "upload returned no file id",
                }

            file_item = {
                "type": "file",
                "id": file_id,
                "name": filename,
                "url": file_id,
                "size": len(data),
                "mime_type": media_type,
                "file": uploaded,
            }
            event = client.post(
                f"{OPENWEBUI_BASE_URL}/api/v1/chats/{chat_id}/messages/{message_id}/event",
                headers={**headers, "Accept": "application/json"},
                json={"type": "files", "data": {"files": [file_item]}},
            )
            event.raise_for_status()
            return {
                "attached_to_chat": True,
                "openwebui_file_id": file_id,
            }
    except Exception as e:
        logger.warning("OpenWebUI file attach failed for %s: %s", filename, e)
        return {
            "attached_to_chat": False,
            "attach_reason": f"OpenWebUI attach failed: {type(e).__name__}",
        }


def _file_result(
    data: bytes,
    media_type: str,
    filename: str,
    request: Optional[Request] = None,
) -> list[Any]:
    b64 = base64.b64encode(data).decode("ascii")
    metadata = {
        "status": "success",
        "filename": filename,
        "mime_type": media_type,
        "bytes": len(data),
    }
    if request is not None:
        metadata.update(_attach_file_to_openwebui(request, data, media_type, filename))
    return [
        f"data:{media_type};base64,{b64}",
        metadata,
    ]


def _is_blocked_ip(ip: str) -> bool:
    parsed = ipaddress.ip_address(ip)
    return (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    )


def _validate_public_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise HTTPException(
            status_code=400,
            detail="url must start with http:// or https://",
        )
    host = parsed.hostname
    if not host:
        raise HTTPException(status_code=400, detail="url must include a hostname")
    host_l = host.lower().rstrip(".")
    if host_l in {"localhost", "host.docker.internal", "docker.for.mac.localhost"}:
        raise HTTPException(
            status_code=400,
            detail="private/internal URLs are not allowed",
        )
    try:
        if _is_blocked_ip(host_l):
            raise HTTPException(
                status_code=400,
                detail="private/internal URLs are not allowed",
            )
        return
    except ValueError:
        pass
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        infos = socket.getaddrinfo(host, port)
    except socket.gaierror as e:
        raise HTTPException(status_code=400, detail=f"could not resolve hostname: {e}")
    for info in infos:
        resolved_ip = info[4][0]
        if _is_blocked_ip(resolved_ip):
            raise HTTPException(
                status_code=400,
                detail="private/internal URLs are not allowed",
            )


def _download_public_url(url: str) -> str:
    current = url
    headers = {"User-Agent": "owui-tool-server/0.3.0"}
    with httpx.Client(timeout=15.0, follow_redirects=False, headers=headers) as client:
        for _ in range(5):
            _validate_public_http_url(current)
            response = client.get(current)
            if 300 <= response.status_code < 400 and response.headers.get("location"):
                current = urljoin(str(response.url), response.headers["location"])
                continue
            response.raise_for_status()
            return response.text
    raise HTTPException(status_code=508, detail="too many redirects")


# --- Endpoints ------------------------------------------------------------

@app.get("/health", summary="Liveness check", operation_id="health")
def health() -> dict:
    return {"status": "ok", "pandoc": pypandoc.get_pandoc_version()}


@app.post(
    "/export/docx",
    summary="Export markdown to a Microsoft Word .docx file",
    description=(
        "Convert APA-formatted (or any) markdown to a Word document. Returns "
        "an OpenWebUI-compatible file payload. Use when the user asks for a Word "
        "document, .docx export, or 'export to Word'. Generate the full final "
        "markdown first, then export that same content."
    ),
    response_description="OWUI file payload for a .docx file",
    operation_id="export_docx",
)
def export_docx(req: ExportRequest, request: Request) -> list[Any]:
    filename = _safe_filename(req.filename, "document")
    extra_args = ["--standalone"]
    if req.title:
        extra_args += ["--metadata", f"title={req.title}"]
    try:
        data = _convert_pandoc(req.markdown, "docx", extra_args)
    except Exception as e:
        logger.exception("docx export failed")
        raise HTTPException(status_code=500, detail=f"docx conversion failed: {e}")
    return _file_result(
        data,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        f"{filename}.docx",
        request,
    )


@app.post(
    "/export/pdf",
    summary="Export markdown to a PDF file",
    description=(
        "Convert markdown to a PDF document. Returns an OpenWebUI-compatible "
        "file payload. Use when the user asks for a PDF, a printable version, or "
        "a read-only share. Generate the full final markdown first, then export "
        "that same content."
    ),
    response_description="OWUI file payload for a PDF file",
    operation_id="export_pdf",
)
def export_pdf(req: ExportRequest, request: Request) -> list[Any]:
    filename = _safe_filename(req.filename, "document")
    # weasyprint = lightweight; xelatex would give publication-grade output
    # but bloats the image by ~3GB.
    extra_args = ["--standalone", "--pdf-engine=weasyprint"]
    if req.title:
        extra_args += ["--metadata", f"title={req.title}"]
    try:
        data = _convert_pandoc(req.markdown, "pdf", extra_args)
    except Exception as e:
        logger.exception("pdf export failed")
        raise HTTPException(status_code=500, detail=f"pdf conversion failed: {e}")
    return _file_result(data, "application/pdf", f"{filename}.pdf", request)


@app.post(
    "/export/markdown",
    summary="Save content as a .md file",
    description=(
        "Save markdown content as a downloadable .md file. Use when the user "
        "asks for a markdown export, a .md file, or wants to download the "
        "raw markdown of a draft. Export the complete content, not a summary."
    ),
    response_description="OWUI file payload for a .md file",
    operation_id="export_markdown",
)
def export_markdown(req: ExportRequest, request: Request) -> list[Any]:
    filename = _safe_filename(req.filename, "document")
    return _file_result(
        req.markdown.encode("utf-8"),
        "text/markdown",
        f"{filename}.md",
        request,
    )


@app.post(
    "/export/csv",
    summary="Export tabular data to a .csv file",
    description=(
        "Convert a list of dictionaries to a CSV file. The keys of the first "
        "row become the column headers. Use when the user asks for CSV, "
        "spreadsheet export, or wants tabular data they can open in Excel."
    ),
    response_description="OWUI file payload for a .csv file",
    operation_id="export_csv",
)
def export_csv(req: CsvExportRequest, request: Request) -> list[Any]:
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows cannot be empty")
    filename = _safe_filename(req.filename, "data")
    fieldnames = list(req.rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(req.rows)
    return _file_result(
        buf.getvalue().encode("utf-8"),
        "text/csv",
        f"{filename}.csv",
        request,
    )


@app.post(
    "/fetch_url",
    summary="Fetch a URL and return its readable text content",
    description=(
        "Download a webpage and extract the main readable text (stripping nav, "
        "ads, boilerplate). Use when the user gives a specific URL to summarise "
        "or quote. Prefer this over web search for exact URL requests; use web "
        "search only when broader discovery is requested."
    ),
    operation_id="fetch_url",
)
def fetch_url(req: FetchUrlRequest) -> dict:
    try:
        downloaded = _download_public_url(req.url)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"fetch failed with HTTP {e.response.status_code}: {req.url}",
        )
    except Exception as e:
        logger.exception("fetch_url failed at download")
        raise HTTPException(status_code=502, detail=f"fetch failed: {e}")
    if not downloaded:
        raise HTTPException(status_code=502, detail=f"could not fetch {req.url}")
    text = trafilatura.extract(downloaded, include_links=True, include_tables=True) or ""
    truncated = False
    if len(text) > req.max_chars:
        text = text[: req.max_chars] + "\n\n[...truncated]"
        truncated = True
    return {
        "url": req.url,
        "chars": len(text),
        "truncated": truncated,
        "text": text,
    }


@app.post(
    "/lookup_doi_citation",
    summary="Look up a DOI on CrossRef and return an APA citation",
    description=(
        "Query CrossRef for a DOI and return the work as an APA-formatted "
        "citation string plus structured metadata (authors, year, title, "
        "journal). Use when the user gives a DOI and wants the formatted "
        "reference, or when verifying a citation they plan to use."
    ),
    operation_id="lookup_doi_citation",
)
def lookup_doi_citation(req: CitationRequest) -> dict:
    doi = req.doi.strip()
    for prefix in ("doi:", "https://doi.org/", "http://doi.org/", "https://dx.doi.org/"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
            break
    try:
        r = httpx.get(
            f"https://api.crossref.org/works/{quote(doi, safe='')}",
            headers={"User-Agent": "owui-tool-server/0.3.0"},
            timeout=15.0,
        )
        r.raise_for_status()
        msg = r.json()["message"]
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=404, detail=f"CrossRef returned {e.response.status_code} for DOI {doi}")
    except Exception as e:
        logger.exception("DOI lookup failed")
        raise HTTPException(status_code=502, detail=f"CrossRef lookup failed: {e}")

    authors = msg.get("author", []) or []
    def _author_str(a: dict) -> str:
        family = a.get("family", "").strip()
        given = a.get("given", "").strip()
        initials = "".join(f"{p[0]}." for p in given.split() if p)
        return f"{family}, {initials}".strip(", ")
    author_parts = [_author_str(a) for a in authors[:20] if a.get("family")]
    if len(authors) > 20:
        author_parts.append("...")
    author_str = ", ".join(author_parts) if author_parts else "Unknown"

    issued = msg.get("issued") or msg.get("published") or {}
    date_parts = issued.get("date-parts") or [[None]]
    year = date_parts[0][0] if date_parts and date_parts[0] else "n.d."

    title = (msg.get("title") or [""])[0]
    container = (msg.get("container-title") or [""])[0]
    volume = msg.get("volume", "")
    issue = msg.get("issue", "")
    pages = msg.get("page", "")
    doi_canonical = msg.get("DOI", doi)

    parts = [f"{author_str} ({year}). {title}."]
    if container:
        loc = f"*{container}*"
        if volume:
            loc += f", *{volume}*"
            if issue:
                loc += f"({issue})"
        if pages:
            loc += f", {pages}"
        parts.append(f"{loc}.")
    parts.append(f"https://doi.org/{doi_canonical}")
    apa = " ".join(parts)

    return {
        "doi": doi_canonical,
        "apa": apa,
        "title": title,
        "year": year,
        "container": container,
        "authors": [{"family": a.get("family"), "given": a.get("given")} for a in authors],
    }
