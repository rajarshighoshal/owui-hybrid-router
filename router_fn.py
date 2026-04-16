"""
title: Advanced Hybrid Router & Interceptor
description: Accuracy-first router. Semantic routing + sticky follow-up + contextual classifier + strict citation enforcement (inlet prompts + outlet regen) + WebUI status emission.
author: rajarshi
version: 9.5
"""

import asyncio
import base64
import hashlib
import logging
import math
import os
import random
import re
import sqlite3
import struct
import time
import uuid
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Optional

import aiohttp
from pydantic import BaseModel, Field

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

EventEmitter = Optional[Callable[[dict], Awaitable[Any]]]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

SEARCH_MARKER = "LIVE WEB SEARCH RESULTS"

FORCE_SEARCH_PATTERNS = [
    re.compile(r"\bsearch\b", re.IGNORECASE),
    re.compile(r"\bweb\s+search\b", re.IGNORECASE),
    re.compile(r"\blook\s+up\b", re.IGNORECASE),
    re.compile(r"\bgoogle\b", re.IGNORECASE),
    re.compile(r"\bfind\s+online\b", re.IGNORECASE),
    re.compile(r"\blatest\b", re.IGNORECASE),
    re.compile(r"\bcurrent(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\bnowadays\b", re.IGNORECASE),
    re.compile(r"\brecently\b", re.IGNORECASE),
    re.compile(r"\bas\s+of\b", re.IGNORECASE),
    re.compile(r"\bup\s+to\s+date\b", re.IGNORECASE),
    re.compile(r"\btoday\b", re.IGNORECASE),
    re.compile(r"\bthis\s+(year|month|week)\b", re.IGNORECASE),
    re.compile(r"\bbreaking\b", re.IGNORECASE),
    re.compile(r"\bnews\s+about\b", re.IGNORECASE),
    re.compile(r"\bupdate\s+on\b", re.IGNORECASE),
    re.compile(r"\bwhat'?s\s+the\s+latest\b", re.IGNORECASE),
]

CATEGORY_NAMES = frozenset({"FACTUAL", "REASONING", "CODING", "RESEARCH", "CASUAL"})

# Models confirmed to natively accept image_url content parts on Fireworks.
# All other models will have images replaced with text captions (vision proxy).
# Updated 2026-04-15 via live API testing — Fireworks metadata is unreliable.
VISION_CAPABLE_MODELS = frozenset(
    {
        "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct",
        "accounts/fireworks/models/kimi-k2p5",
        "accounts/fireworks/models/deepseek-v3p1",
    }
)

# Fallback model chains — when the primary model is down (503, timeout),
# try the next one in the chain. Ordered by cost (cheapest first for
# classifier/verifier, best quality first for main/caption).
# Updated 2026-04-16 via live API testing.
CLASSIFIER_FALLBACK_CHAIN = [
    "accounts/fireworks/models/mixtral-8x22b-instruct",  # $1.20/1M — primary, great at classification
    "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct",  # $0.15/$0.60 — cheap, returns just the category name
    "accounts/fireworks/models/gpt-oss-20b",  # $0.07/$0.30 — cheapest, reasons first but works
]

VERIFIER_FALLBACK_CHAIN = [
    "accounts/fireworks/models/mixtral-8x22b-instruct",  # $1.20/1M — primary, follows output format instructions
    "accounts/fireworks/models/deepseek-v3p1",  # $0.56/$1.68 — good instruction following
    "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct",  # $0.15/$0.60 — last resort
]

CAPTION_FALLBACK_CHAIN = [
    "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct",  # $0.15/$0.60 — primary, accurate concise captions
    "accounts/fireworks/models/kimi-k2p5",  # $0.60/$3.00 — works but verbose and expensive
]

THINKING_BLOCK_RE = re.compile(
    r"<think>.*?</think>|<tool_call>.*?</tool_call>", re.DOTALL
)

_FOLLOWUP_STARTERS = re.compile(
    r"^\s*(and|also|now|but|so|or|plus|what\s+about|how\s+about|what\s+if|yeah|yes|ok|okay|more|again)\b",
    re.IGNORECASE,
)
_PRONOUN_REF = re.compile(
    r"\b(it|its|it's|that|this|these|those|they|them|their|he|she|him|her|hers|his)\b",
    re.IGNORECASE,
)

_ROUTE_HEADER_RE = re.compile(
    r"\A`[^\n`]*\b(?:FACTUAL|REASONING|CODING|RESEARCH|CASUAL)\b[^\n`]*`\s*\n(?:>\s*[^\n]*\n)?\s*",
    re.IGNORECASE,
)

CLAIM_PATTERNS = re.compile(
    r"\$\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|trillion|k|m|b))?"
    r"|\d+(?:\.\d+)?\s*%"
    r"|\b(?:in|during|by|since|as\s+of|before|after|until|from|through|circa)\s+(?:19|20)\d{2}\b"
    r"|\b\d+(?:\.\d+)?\s*(?:million|billion|thousand|trillion|kg|km|miles|meters|feet|USD|EUR|GBP|JPY|people|users|cases|deaths|votes)\b"
    r"|\b(?:about|approximately|roughly|nearly|over|under|more\s+than|less\s+than)\s+\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
CITATION_PATTERNS = re.compile(
    r"https?://\S+" r"|\[\s*(?:Source|URL|arxiv|doi|ref)\b[^\]]*\]" r"|\[\d+\]",
    re.IGNORECASE,
)


def _is_followup_query(query: str) -> bool:
    words = query.split()
    if not words or len(words) > 15:
        return False
    if _FOLLOWUP_STARTERS.match(query):
        return True
    if len(words) <= 6 and _PRONOUN_REF.search(query):
        return True
    return False


class _NonRetryableError(Exception):
    pass


def _truncate_at_sentence(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    last_period = chunk.rfind(".")
    last_newline = chunk.rfind("\n")
    cut = max(last_period, last_newline)
    return chunk[: cut + 1] if cut > max_chars // 2 else chunk


_INJECTION_RE = re.compile(
    r"(?i)\b("
    r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?|messages?)"
    r"|disregard\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?|messages?)"
    r"|forget\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|messages?)"
    r"|new\s+instructions?\s*:"
    r"|system\s+prompt\s*:"
    r")\b.*"
)


def _sanitize_query(query: str) -> str:
    sanitized = query.replace("```", "")
    sanitized = _INJECTION_RE.sub("[redacted]", sanitized)
    return sanitized.strip()


def _extract_text(content) -> str:
    """Extract plain text from message content that may be a string or a
    list of content-parts (vision+chat format).

    OpenAI multimodal format uses:
        [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
    This helper always returns a string, joining all text parts.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(parts)
    return str(content)


def _extract_images(content) -> list[dict]:
    """Extract image_url content parts from a multimodal message.

    Returns a list of {"type": "image_url", "image_url": {"url": ...}} dicts
    ready to be forwarded directly to a vision model API.
    Returns an empty list for plain-text messages.
    """
    if not isinstance(content, list):
        return []
    return [
        part
        for part in content
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]


def _strip_thinking_blocks(text: str) -> str:
    return THINKING_BLOCK_RE.sub("", text)


def _is_retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


async def _retry_request(coro_fn, max_retries: int = 2, base_delay: float = 0.5):
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except _NonRetryableError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                "Retry %d/%d after error: %s — waiting %.1fs",
                attempt + 1,
                max_retries,
                e,
                delay,
            )
            await asyncio.sleep(delay)


def _cosine_similarity(v1: list, v2: list) -> float:
    if _HAS_NUMPY:
        a = np.asarray(v1, dtype=np.float64)
        b = np.asarray(v2, dtype=np.float64)
        n1 = np.linalg.norm(a)
        n2 = np.linalg.norm(b)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(a, b) / (n1 * n2))
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return 0.0 if norm1 == 0 or norm2 == 0 else dot / (norm1 * norm2)


# ---------------------------------------------------------------------------
# Chat-memory helpers (Phase 1). Purely module-level — no per-chat state here.
# ---------------------------------------------------------------------------

# Acknowledgment-only messages have no retrieval value — skip them on store.
_ACK_ONLY_RE = re.compile(
    r"^\s*(ok(ay)?|yes|yep|yeah|no|nope|thanks?|thank\s*you|cool|"
    r"got\s*it|sure|alright|fine|right|hmm+|uh+|mm+)[\s!.,?]*$",
    re.IGNORECASE,
)

# Verification trailer appended by the outlet when citations fail — strip
# before storing so the memory entry is just the answer, not the meta.
_VERIFICATION_TRAILER_RE = re.compile(
    r"\n\n---\n⚠️\s*\*\*Verification note:.*?(?=\n*$|\Z)",
    re.DOTALL,
)


def _f32_pack(vec: list) -> bytes:
    """Serialize a float list to raw float32 bytes for SQLite BLOB storage."""
    return struct.pack(f"{len(vec)}f", *vec)


def _f32_unpack(blob: bytes) -> list:
    """Deserialize float32 bytes back to a Python list."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def _memory_content_hash(text: str) -> str:
    """SHA-256 of normalized text. Used for chat-scoped dedup."""
    return hashlib.sha256(text.strip().encode("utf-8", errors="replace")).hexdigest()


def _clean_for_memory(text: str) -> str:
    """Strip mechanical artifacts before storing a turn as memory.

    Removes (in order): thinking blocks, ROUTER_STATE tags, the route-tag
    header, and the verifier's UNVERIFIED trailer. Leaves the actual
    user-visible answer intact.
    """
    if not text:
        return ""
    text = THINKING_BLOCK_RE.sub("", text)
    text = re.sub(r"\[ROUTER_STATE:\s*[A-Z]+(?:_SEARCH)?\]", "", text)
    text = _ROUTE_HEADER_RE.sub("", text, count=1)
    text = _VERIFICATION_TRAILER_RE.sub("", text)
    return text.strip()


async def _check_response(resp: aiohttp.ClientResponse) -> None:
    if resp.status == 200:
        return
    text = await resp.text()
    err = aiohttp.ClientResponseError(
        request_info=resp.request_info,
        history=resp.history,
        status=resp.status,
        message=f"HTTP {resp.status}: {text[:200]}",
    )
    if _is_retryable_status(resp.status):
        raise err
    raise _NonRetryableError(str(err)) from err


class Filter:
    class Valves(BaseModel):
        FIREWORKS_API_KEY: str = Field(
            default="", description="Your Fireworks.ai API Key."
        )
        TAVILY_API_KEY: str = Field(
            default="", description="Your Tavily API Key for web search."
        )
        EMBEDDING_MODEL: str = Field(
            default="nomic-ai/nomic-embed-text-v1.5", description="Intent vectors."
        )
        CLASSIFIER_MODEL: str = Field(
            default="accounts/fireworks/models/mixtral-8x22b-instruct",
            description="Routing model. Must be present on your Fireworks account. "
            "mixtral is the cheapest catalog option and handles intent classification fine.",
        )
        MAIN_MODEL: str = Field(
            default="accounts/fireworks/models/glm-5p1", description="Heavy model."
        )
        VERIFIER_MODEL: str = Field(
            default="accounts/fireworks/models/mixtral-8x22b-instruct",
            description="Citation auditor model. Use an instruction-following model (not a reasoning one) — "
            "the verifier needs to emit a clean PASS/FAIL line, not chain-of-thought. "
            "Mixtral respects output-format instructions more reliably than deepseek/glm for this task.",
        )
        ROUTING_THRESHOLD: float = Field(
            default=0.6,
            description="Cosine similarity threshold. Below this, fall back to the LLM classifier. "
            "0.85 is too strict for nomic-embed against short category descriptions; "
            "0.55–0.7 gives the embedding path a chance to actually fire.",
        )
        SHOW_ROUTE_TAG: bool = Field(
            default=True, description="Show the route category tag."
        )
        ENABLE_OUTLET_VERIFICATION: bool = Field(
            default=True,
            description="Run a citation check on FACTUAL/RESEARCH responses. "
            "If verification fails, behavior is controlled by VERIFIER_REGENERATE.",
        )
        VERIFIER_MODE: str = Field(
            default="hybrid",
            description="'regex' (cheap structural check), 'llm' (faithfulness audit via classifier model), "
            "or 'hybrid' (regex gate first, then LLM deep-check). "
            "'hybrid' catches uncited claims AND hallucinated/mis-attributed URLs.",
        )
        VERIFIER_REGENERATE: bool = Field(
            default=False,
            description="If verification fails, REGENERATE the response (costly, can degrade quality). "
            "When False (default), a warning banner is appended to the original response instead.",
        )
        VERIFIER_MAX_TOKENS: int = Field(
            default=2500,
            description="Max output tokens for the verifier regeneration. High default — accuracy over cost.",
        )
        VERIFIER_MAX_RETRIES: int = Field(
            default=2,
            description="How many times the verifier will regenerate + re-verify after a FAIL. "
            "Each attempt appends its corrected version via event emitter; the first passing attempt wins. "
            "If all attempts fail, the last one is still shown with an 'UNVERIFIED' banner.",
        )
        OUTLET_VERIFY_TIMEOUT: int = Field(
            default=300,
            description="Cap on the full outlet verification + regeneration loop in seconds. "
            "Generous default (5 min) — accuracy over latency.",
        )
        SEARCH_RESULTS_FACTUAL: int = Field(
            default=6,
            description="Tavily results requested for FACTUAL routes. Higher = more grounding.",
        )
        SEARCH_RESULTS_RESEARCH: int = Field(
            default=10,
            description="Tavily results requested for RESEARCH routes. Higher = broader literature coverage.",
        )
        SEARCH_CACHE_TTL: int = Field(
            default=3600, description="Search cache TTL in seconds."
        )
        SEARCH_CACHE_MAX: int = Field(
            default=100, description="Max entries in search cache."
        )
        INLET_TIMEOUT: int = Field(
            default=90,
            description="Overall inlet timeout in seconds. Covers anchor init + query embedding + "
            "classifier LLM + Tavily search. On timeout the router passes the query through unrouted "
            "so the main model still answers.",
        )
        EMIT_STATUS_EVENTS: bool = Field(
            default=True,
            description="Emit critical status events to the OpenWebUI conversation surface (search failures, "
            "router fallbacks, citation regeneration).",
        )
        ENABLE_STICKY_ROUTING: bool = Field(
            default=True,
            description="Carry the last category + search-flag forward for short/referential follow-ups "
            "(e.g., 'and the EU version?'). Only fires when embedding confidence is below threshold.",
        )
        STICKY_MAX_CONVOS: int = Field(
            default=200, description="Max chats retained in the sticky-route LRU."
        )
        STICKY_TTL_SECONDS: int = Field(
            default=86400,
            description="Sticky route entries expire after this many seconds (default 24h). "
            "Prevents a week-old chat from inheriting stale routing.",
        )
        ENABLE_IMAGE_ROUTING: bool = Field(
            default=True,
            description="When a message contains images, call a vision model to generate a short "
            "caption. The caption is injected into the routing query (embedding + classifier) and "
            "the Tavily search query — never sent to the main model. Has no effect on text-only messages.",
        )
        IMAGE_CAPTION_MODEL: str = Field(
            default="accounts/fireworks/models/qwen3-vl-30b-a3b-instruct",
            description="Vision model for captioning images during routing. Must be serverless "
            "on your Fireworks account and accept image_url content parts. "
            "Qwen3 VL 30B A3B ($0.15/$0.60 per 1M tokens) is the cheapest serverless vision model "
            "on Fireworks — MoE architecture means only 3B params are active per token, so it's fast "
            "and cheap for short captions. kimi-k2p5 also works with images but costs 4x more. "
            "NOTE: Most Fireworks vision models (llama-v3p2-11b, qwen2p5-vl-7b) require dedicated GPU, "
            "NOT serverless — they return 404.",
        )
        IMAGE_CAPTION_MAX_TOKENS: int = Field(
            default=80,
            description="Max output tokens for the ROUTING caption (short, 1-2 sentences). "
            "Keep low — this is just for routing classification, not for the main model.",
        )
        ENABLE_VISION_PROXY: bool = Field(
            default=True,
            description="When a non-vision model (e.g. GLM-5.1, DeepSeek V3.2) receives a message "
            "with images, automatically caption each image with a rich description and replace "
            "the image parts with text so the model can 'see' it. Runs on EVERY turn to handle "
            "images in conversation history too. Captions are cached per chat so follow-up turns "
            "don't re-caption the same images. Vision-capable models (Qwen3 VL, Kimi K2.5) "
            "always receive images natively — the proxy is a no-op for them.",
        )
        IMAGE_PROXY_MAX_TOKENS: int = Field(
            default=300,
            description="Max output tokens for the DETAILED caption used by the vision proxy. "
            "This caption is what the main model will 'see' instead of the image — it needs to be "
            "rich enough to preserve key details: visible text, error messages, code, chart labels, "
            "layout, colors, numbers. 300 tokens ≈ 2-3 paragraphs, enough for a complex screenshot.",
        )
        ENABLE_CHAT_MEMORY: bool = Field(
            default=True,
            description="Persistent per-chat semantic memory. When a chat grows beyond "
            "CHAT_MEMORY_MIN_TURNS, the router embeds the current query and injects the "
            "top-K most similar prior turns (from THIS chat only) into the system prompt. "
            "Strictly chat-scoped — the only filter on every query is WHERE chat_id=?. "
            "Never leaks across chats or users.",
        )
        CHAT_MEMORY_DB_PATH: str = Field(
            default="/app/backend/data/router_mem.db",
            description="SQLite file for chat memory. Lives in OpenWebUI's persistent data "
            "volume so it survives container recreation via deploy.sh's volume inheritance.",
        )
        CHAT_MEMORY_MIN_TURNS: int = Field(
            default=15,
            description="Only recall memory for chats with more than N total turns stored. "
            "Below this, OpenWebUI's own chat-history injection is sufficient context.",
        )
        CHAT_MEMORY_TOP_K: int = Field(
            default=8,
            description="Number of most-similar prior turns to inject when recalling memory. "
            "Higher = more recalled context, more input tokens.",
        )
        CHAT_MEMORY_TTL_DAYS: int = Field(
            default=90,
            description="Prune chat memory rows older than this many days. Ran probabilistically "
            "on ~1%% of outlet calls along with the referential sweep.",
        )
        CHAT_MEMORY_MAX_TURNS_PER_CHAT: int = Field(
            default=500,
            description="Hard cap on stored turns per chat. Oldest are dropped when exceeded. "
            "Prevents a single runaway chat from eating disk space.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.anchor_embeddings: dict[str, list] = {}
        self._last_embedding_model: Optional[str] = None
        self.search_cache: OrderedDict[str, dict] = OrderedDict()
        self._last_tavily_key: Optional[str] = None
        self.sticky_routes: OrderedDict[str, dict] = OrderedDict()
        self._session_affinity_id = uuid.uuid4().hex
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock: Optional[asyncio.Lock] = None
        self._embedding_lock: Optional[asyncio.Lock] = None
        # Cache: (chat_id, image_url_or_hash) → detailed caption.
        # Prevents re-captioning the same image on follow-up turns.
        self.image_caption_cache: OrderedDict[str, str] = OrderedDict()
        self._owui_base_url: Optional[str] = None  # auto-detected, cached
        # Chat memory state (Phase 1). Lazy-initialized; disabled sticky on
        # any DB error so broken memory never breaks a reply.
        self._memory_conn: Optional[sqlite3.Connection] = None
        self._memory_conn_lock: Optional[asyncio.Lock] = None
        self._memory_disabled: bool = False

        self.categories = {
            "FACTUAL": "Objective inquiries requiring real-world verification. Focus on data, laws, prices, and current events.",
            "REASONING": "Mathematical, logical, and algorithmic reasoning. Focus on step-by-step proofs and theoretical math.",
            "CODING": "Programming, software engineering, and code execution. Focus on code generation and syntax.",
            "RESEARCH": "Academic literature, paper analysis, and scientific consensus. Focus on citations and academia.",
            "CASUAL": "Subjective, creative, or conversational interactions. Focus on opinions and greetings without factual constraints.",
        }

        self.prompts = {
            "FACTUAL": (
                "MANDATORY: Use the WEB SEARCH RESULTS provided below to answer.\n"
                "1. Treat retrieved text as the only truth. Do not use prior knowledge that contradicts the results.\n"
                "2. CITATION FORMAT IS STRICT. Inline citations MUST be numeric refs only: [1], [2], etc. "
                "FORBIDDEN inline: any URL, any [Source: ...] bracket, any title/author/year metadata. "
                "URLs appear ONLY in the Sources section at the bottom.\n"
                "3. Every factual claim (numbers, dates, percentages, dollar amounts, names, locations) MUST carry a [N] ref.\n"
                "4. End the response with EXACTLY this structure:\n"
                "   ---\n"
                "   **Sources:**\n"
                "   [1] <full URL from SEARCH_RESULTS>\n"
                "   [2] <full URL from SEARCH_RESULTS>\n"
                "5. Use [DATA NOT FOUND] verbatim if the specific fact is missing. Do not guess or hallucinate.\n"
                "6. Use bullet points. Bold key terms. Place the [N] ref immediately after each claim.\n\n"
                "EXAMPLE SHAPE (abstract — URLs are ONLY at the bottom):\n"
                "- First factual claim about the topic [1].\n"
                "- Second claim citing a different source [2].\n"
                "- Another claim from the first source [1].\n"
                "\n"
                "---\n"
                "**Sources:**\n"
                "[1] <full url from SEARCH_RESULTS>\n"
                "[2] <full url from SEARCH_RESULTS>"
            ),
            "REASONING": (
                "REASONING MODE. Think step-by-step. Be mathematically rigorous. "
                "State assumptions explicitly. Prove complexity bounds. "
                "If a step is non-trivial, show the derivation. Do not skip algebra."
            ),
            "CODING": (
                "CODE MODE. Write correct, minimal, runnable code. "
                "No unnecessary abstractions. Comments explain WHY not WHAT. "
                "Include imports. Note language/runtime version when relevant. "
                "If the request has edge cases (empty input, overflow, concurrency), address them."
            ),
            "RESEARCH": (
                "RESEARCH MODE. Use the WEB SEARCH RESULTS below to answer.\n"
                "CITATION FORMAT IS STRICT. Inline citations MUST be numeric refs only: [1], [2], etc. "
                "FORBIDDEN inline: any URL, any [Source: ...] bracket, any (title, author, year) parenthetical.\n"
                "URLs and source metadata appear ONLY in the Sources section at the bottom.\n"
                "Every claim MUST carry a [N] ref. Uncited claims are errors.\n\n"
                "End the response with EXACTLY this structure:\n"
                "   ---\n"
                "   **Sources:**\n"
                "   [1] <entry>\n"
                "   [2] <entry>\n"
                "Sources-section entries by source type:\n"
                '• Academic paper: `[N] Author et al., Year, "Title" — <full URL>`\n'
                "• News / blog / web: `[N] <full URL>`\n"
                "• Tavily aggregate: `[N] Tavily AI Summary` (only valid if SEARCH_RESULTS has a 'Tavily AI Summary' section)\n"
                "The URL must appear verbatim in SEARCH_RESULTS. Do not fabricate citations or invent paper titles.\n\n"
                "SOURCE HIERARCHY for academic queries: arxiv.org > peer-reviewed journals > conference proceedings > preprints > blogs.\n"
                "FORMAT: Use ### Headers for distinct topics. Use bullet points for findings. "
                "Place the [N] ref immediately after each claim.\n\n"
                "EXAMPLE SHAPE (abstract):\n"
                "### First topic\n"
                "- A claim about the topic [1].\n"
                "- Another claim, different source [2].\n"
                "\n"
                "### Second topic\n"
                "- A further claim from the first source [1].\n"
                "\n"
                "---\n"
                "**Sources:**\n"
                "[1] <full url from SEARCH_RESULTS>\n"
                '[2] <Author, Year, "Title"> — <full url from SEARCH_RESULTS>'
            ),
            "CASUAL": "You are a helpful assistant. Be warm, witty, and direct. Have opinions. Show personality.",
        }

    def __del__(self):
        if self._session and not self._session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._session.close())
                else:
                    loop.run_until_complete(self._session.close())
            except Exception:
                pass

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    on_shutdown = close

    def _get_embedding_lock(self) -> asyncio.Lock:
        if self._embedding_lock is None:
            self._embedding_lock = asyncio.Lock()
        return self._embedding_lock

    def _get_session_lock(self) -> asyncio.Lock:
        if self._session_lock is None:
            self._session_lock = asyncio.Lock()
        return self._session_lock

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is not None and not self._session.closed:
            return self._session
        async with self._get_session_lock():
            if self._session is not None and not self._session.closed:
                return self._session
            self._session = aiohttp.ClientSession()
            return self._session

    def _invalidate_stale_caches(self) -> None:
        if self.valves.EMBEDDING_MODEL != self._last_embedding_model:
            self.anchor_embeddings.clear()
            self._last_embedding_model = self.valves.EMBEDDING_MODEL
            logger.info("Embedding model changed — cleared anchor cache.")
        if self.valves.TAVILY_API_KEY != self._last_tavily_key:
            self.search_cache.clear()
            self._last_tavily_key = self.valves.TAVILY_API_KEY
            logger.info("Tavily API key changed — cleared search cache.")

    async def _search_tavily(self, query: str, max_results: int = 4) -> str:
        if not self.valves.TAVILY_API_KEY:
            return "[No Tavily API Key Provided]"
        # Hash the query so raw user text never sits in RAM as a dict key.
        # Content is public Tavily data, but the KEY preserved the original
        # query verbatim for up to SEARCH_CACHE_TTL. On a shared family server
        # that's a weak privacy smell — the hash removes it without changing
        # cache semantics.
        cache_key = hashlib.sha256(
            f"{query.strip().lower()}_{max_results}".encode()
        ).hexdigest()
        if cache_key in self.search_cache:
            entry = self.search_cache[cache_key]
            if time.time() - entry["timestamp"] < self.valves.SEARCH_CACHE_TTL:
                self.search_cache.move_to_end(cache_key)
                return entry["results"]
            del self.search_cache[cache_key]

        async def _do_search():
            session = await self._get_session()
            async with session.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.valves.TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "max_results": max_results,
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                await _check_response(resp)
                data = await resp.json()
                context = f"Tavily AI Summary: {data.get('answer', '')}\n\n"
                for res in data.get("results", []):
                    context += f"Source ({res.get('url')}):\n{res.get('content')}\n\n"
                return _truncate_at_sentence(context)

        try:
            result = await _retry_request(_do_search)
            self.search_cache[cache_key] = {
                "results": result,
                "timestamp": time.time(),
            }
            self.search_cache.move_to_end(cache_key)
            while len(self.search_cache) > self.valves.SEARCH_CACHE_MAX:
                self.search_cache.popitem(last=False)
            return result
        except _NonRetryableError as e:
            logger.warning("Tavily non-retryable error: %s", e)
            return f"[Web Search Failed: {e}]"
        except Exception as e:
            logger.warning("Tavily search failed after retries: %s", e)
            return f"[Web Search Failed: {e}]"

    async def _get_embedding(self, text: str) -> list:
        try:

            async def _do_embed():
                session = await self._get_session()
                async with session.post(
                    "https://api.fireworks.ai/inference/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.valves.FIREWORKS_API_KEY}"
                    },
                    json={"model": self.valves.EMBEDDING_MODEL, "input": text},
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    await _check_response(resp)
                    data = await resp.json()
                    return data["data"][0]["embedding"]

            return await _retry_request(_do_embed)
        except _NonRetryableError as e:
            logger.warning("Embedding non-retryable error for '%s…': %s", text[:50], e)
            return []
        except Exception as e:
            logger.warning("Embedding failed after retries for '%s…': %s", text[:50], e)
            return []

    async def _call_llm(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 50,
        fallback_chain: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Call an LLM with automatic fallback to alternative models.

        If the primary model fails (503, timeout, not found), tries each
        model in the fallback_chain in order. Returns the first successful
        response, or None if all models fail.
        """
        models_to_try = [model]
        if fallback_chain:
            # Add fallbacks that aren't already the primary
            for fb in fallback_chain:
                if fb != model and fb not in models_to_try:
                    models_to_try.append(fb)

        last_err = None
        for i, m in enumerate(models_to_try):
            is_fallback = i > 0
            try:

                async def _do_call(model_name=m):
                    session = await self._get_session()
                    async with session.post(
                        "https://api.fireworks.ai/inference/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.valves.FIREWORKS_API_KEY}",
                            "Content-Type": "application/json",
                            "x-session-affinity": self._session_affinity_id,
                        },
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": 0.0,
                        },
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        await _check_response(resp)
                        data = await resp.json()
                        choice = data["choices"][0]
                        if choice.get("finish_reason") == "length":
                            logger.warning(
                                "LLM response truncated (finish_reason=length, model=%s)",
                                model_name,
                            )
                        return choice["message"]["content"].strip()

                result = await _retry_request(_do_call)
                if is_fallback:
                    logger.info(
                        "LLM fallback succeeded: %s (primary %s failed)",
                        m.split("/")[-1],
                        model.split("/")[-1],
                    )
                return result
            except _NonRetryableError as e:
                last_err = e
                logger.warning(
                    "LLM non-retryable error (model=%s): %s — trying next fallback",
                    m.split("/")[-1],
                    str(e)[:100],
                )
                continue
            except Exception as e:
                last_err = e
                logger.warning(
                    "LLM call failed after retries (model=%s): %s — trying next fallback",
                    m.split("/")[-1],
                    str(e)[:100],
                )
                continue

        logger.error(
            "All LLM models failed for prompt '%s…' (tried %s)",
            prompt[:50],
            [m.split("/")[-1] for m in models_to_try],
        )
        return None

    async def _ensure_anchor_embeddings(self) -> None:
        if self.anchor_embeddings:
            return
        async with self._get_embedding_lock():
            if self.anchor_embeddings:
                return
            tasks = [self._get_embedding(desc) for desc in self.categories.values()]
            results = await asyncio.gather(*tasks)
            for (cat, _), vec in zip(self.categories.items(), results):
                if vec:
                    self.anchor_embeddings[cat] = vec

    def _parse_llm_category(self, llm_response: Optional[str]) -> Optional[str]:
        if not llm_response:
            return None
        upper = llm_response.upper()
        for cat in CATEGORY_NAMES:
            if re.search(rf"\b{cat}\b", upper):
                return cat
        return None

    @staticmethod
    def _extract_chat_id(body: dict) -> Optional[str]:
        cid = body.get("chat_id") or body.get("id")
        if cid:
            return str(cid)
        meta = body.get("metadata") or {}
        cid = meta.get("chat_id") or meta.get("id")
        return str(cid) if cid else None

    async def _detect_owui_base_url(self) -> Optional[str]:
        """Auto-detect the OpenWebUI base URL for fetching internal images.

        Checks (in order):
        1. Cached result from previous detection
        2. WEBUI_URL env var (set by OpenWebUI itself)
        3. PORT / OPEN_WEBUI_PORT env vars → http://localhost:{port}
        4. Probing common localhost ports (8080, 3000, 80)
        5. Docker container names (open-webui:8080)

        Result is cached after first successful detection.
        """
        if self._owui_base_url is not None:
            return self._owui_base_url or None

        # 1. WEBUI_URL — OpenWebUI sets this for its own API callbacks
        env_url = os.environ.get("WEBUI_URL", "")
        if env_url and env_url.startswith("http"):
            self._owui_base_url = env_url.rstrip("/")
            logger.info(
                "Auto-detected OWUI base URL from WEBUI_URL: %s", self._owui_base_url
            )
            return self._owui_base_url

        # 2. Port from env vars
        for env_key in ("OPEN_WEBUI_PORT", "PORT", "SERVER_PORT"):
            port = os.environ.get(env_key, "")
            if port.isdigit():
                self._owui_base_url = f"http://localhost:{port}"
                logger.info("Auto-detected OWUI base URL from %s", env_key)
                return self._owui_base_url

        # 3. Probe common localhost ports. 11434 is ollama's port and must not
        # be probed here — if ollama returns any accepted status on /api/v1/auths
        # we would misidentify it as OWUI.
        session = await self._get_session()
        for port in (8080, 3000, 80):
            candidate = f"http://localhost:{port}"
            try:
                async with session.head(
                    f"{candidate}/api/v1/auths",
                    timeout=aiohttp.ClientTimeout(total=1),
                ) as probe:
                    if probe.status in (200, 401, 403, 405, 422):
                        self._owui_base_url = candidate
                        logger.info(
                            "Auto-detected OWUI at %s (probe %d)",
                            candidate,
                            probe.status,
                        )
                        return self._owui_base_url
            except Exception:
                continue

        # 4. Docker container name — common default
        for host in ("http://open-webui:8080", "http://openwebui:8080"):
            try:
                async with session.head(
                    f"{host}/api/v1/auths",
                    timeout=aiohttp.ClientTimeout(total=1),
                ) as probe:
                    if probe.status in (200, 401, 403, 405, 422):
                        self._owui_base_url = host
                        logger.info("Auto-detected OWUI at %s", host)
                        return self._owui_base_url
            except Exception:
                continue

        logger.warning(
            "Could not auto-detect OpenWebUI base URL — internal images will not be resolved"
        )
        self._owui_base_url = ""  # cache the failure so we don't re-probe every request
        return None

    async def _resolve_image_urls(
        self,
        image_parts: list[dict],
        event_emitter: EventEmitter = None,
    ) -> list[dict]:
        """Convert image URL parts to base64 data URIs when needed.

        OpenWebUI sends images as internal URLs (e.g. /api/v1/files/...) that
        Fireworks can't fetch. This downloads them locally and rewrites to
        data:image/...;base64,... URIs.  Errors are emitted as chat status events.
        """
        base = await self._detect_owui_base_url()
        resolved: list[dict] = []
        errors: list[str] = []
        session = await self._get_session()

        for part in image_parts:
            url = (part.get("image_url") or {}).get("url", "")

            # Already a data URI — nothing to do
            if url.startswith("data:image/"):
                resolved.append(part)
                continue

            # Public URL — Fireworks can fetch it directly
            if (
                url.startswith("https://")
                and "localhost" not in url
                and "127.0.0.1" not in url
            ):
                resolved.append(part)
                continue

            # Internal / relative URL — need to fetch and convert to base64
            if not base:
                errors.append(
                    f"Internal image URL but OWUI base URL not detected — skipped: {url[:60]}"
                )
                continue

            if url.startswith("/"):
                full_url = f"{base}{url}"
            elif url.startswith("http://localhost") or url.startswith(
                "http://127.0.0.1"
            ):
                path = url.split("//", 1)[-1]
                path = "/" + path.split("/", 1)[-1] if "/" in path else url
                full_url = f"{base}{path}"
            else:
                logger.warning(
                    "Unclassifiable image URL, passing through: %s", url[:100]
                )
                resolved.append(part)
                continue

            # Download and convert to base64 data URI
            try:
                async with session.get(
                    full_url,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as img_resp:
                    if img_resp.status != 200:
                        errors.append(f"Image fetch HTTP {img_resp.status}: {url[:60]}")
                        continue
                    img_bytes = await img_resp.read()
                    content_type = img_resp.headers.get("Content-Type", "image/png")
                    if not content_type.startswith("image/"):
                        if img_bytes[:4] == b"\x89PNG":
                            content_type = "image/png"
                        elif img_bytes[:3] == b"\xff\xd8\xff":
                            content_type = "image/jpeg"
                        elif img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
                            content_type = "image/webp"
                        elif img_bytes[:4] == b"GIF8":
                            content_type = "image/gif"
                        else:
                            content_type = "image/png"
                    b64 = base64.b64encode(img_bytes).decode("ascii")
                    data_uri = f"data:{content_type};base64,{b64}"
                    resolved.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        }
                    )
                    logger.info(
                        "Resolved local image → base64 (%d bytes)", len(img_bytes)
                    )
            except Exception as e:
                errors.append(f"Image fetch error: {str(e)[:80]}")

        # Emit resolution errors to chat
        if errors and event_emitter and self.valves.EMIT_STATUS_EVENTS:
            for err in errors[:3]:
                await self._emit_status(event_emitter, f"⚠️ Image resolve: {err}")

        return resolved

    async def _call_vision_model(
        self,
        vision_content: list[dict],
        max_tokens: int,
        event_emitter: EventEmitter = None,
    ) -> Optional[str]:
        """Call a vision model with automatic fallback across the caption chain.

        Tries each model in CAPTION_FALLBACK_CHAIN until one succeeds.
        Returns the response text, or None if all fail.
        """
        models_to_try = CAPTION_FALLBACK_CHAIN[:]
        # Ensure the configured model is first
        if self.valves.IMAGE_CAPTION_MODEL in models_to_try:
            models_to_try.remove(self.valves.IMAGE_CAPTION_MODEL)
        models_to_try.insert(0, self.valves.IMAGE_CAPTION_MODEL)

        last_err = None
        for i, model_name in enumerate(models_to_try):
            is_fallback = i > 0
            try:

                async def _do_vision_call(mn=model_name):
                    session = await self._get_session()
                    async with session.post(
                        "https://api.fireworks.ai/inference/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.valves.FIREWORKS_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": mn,
                            "messages": [{"role": "user", "content": vision_content}],
                            "max_tokens": max_tokens,
                            "temperature": 0.0,
                        },
                        timeout=aiohttp.ClientTimeout(total=20),
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            err_msg = f"HTTP {resp.status}: {text[:200]}"
                            logger.warning(
                                "Vision API error (model=%s): %s", mn, err_msg
                            )
                            if _is_retryable_status(resp.status):
                                raise aiohttp.ClientResponseError(
                                    request_info=resp.request_info,
                                    history=resp.history,
                                    status=resp.status,
                                    message=err_msg,
                                )
                            raise _NonRetryableError(err_msg)
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"].strip()

                result = await _retry_request(_do_vision_call)
                if is_fallback and event_emitter:
                    await self._emit_status(
                        event_emitter,
                        f"🖼️ Caption fallback: using {model_name.split('/')[-1]} (primary was down).",
                    )
                return result
            except _NonRetryableError as e:
                last_err = str(e)[:150]
                logger.warning(
                    "Vision model non-retryable error (%s): %s",
                    model_name.split("/")[-1],
                    last_err,
                )
                continue
            except Exception as e:
                last_err = str(e)[:150]
                logger.warning(
                    "Vision model failed after retries (%s): %s",
                    model_name.split("/")[-1],
                    last_err,
                )
                continue

        return None

    async def _caption_images(
        self,
        image_parts: list[dict],
        user_text: str = "",
        event_emitter: EventEmitter = None,
    ) -> Optional[str]:
        """Call a vision model to generate a short caption of the images.

        Handles: resolve internal URLs → build prompt → call vision model.
        All errors are emitted as chat status events.
        Returns the caption string on success, or None on failure.
        """
        if not image_parts or not self.valves.IMAGE_CAPTION_MODEL:
            await self._emit_status(
                event_emitter,
                "⚠️ Image captioning: no model configured or no images found.",
            )
            return None

        # Step 1: Resolve internal/local URLs to base64 data URIs
        clean_parts = await self._resolve_image_urls(image_parts, event_emitter)
        if not clean_parts:
            await self._emit_status(
                event_emitter,
                "⚠️ Image captioning: no reachable images after resolving URLs.",
            )
            return None

        # Step 2: Build the vision prompt
        prompt_text = (
            "Describe what you see in this image in 1-2 short sentences. "
            "Focus on: subject matter (code, math, chart, photo, diagram, etc.), "
            "any visible text or labels, and the overall topic."
        )
        if user_text.strip():
            prompt_text += (
                f" The user's accompanying text is: '{user_text.strip()[:200]}'"
            )

        vision_content: list[dict] = [{"type": "text", "text": prompt_text}]
        vision_content.extend(clean_parts)

        # Step 3: Call the vision model with fallback chain
        caption = await self._call_vision_model(
            vision_content,
            self.valves.IMAGE_CAPTION_MAX_TOKENS,
            event_emitter=event_emitter,
        )
        if caption is None:
            await self._emit_status(
                event_emitter,
                "⚠️ Image captioning failed (all models in fallback chain).",
            )
        return caption

    async def _detailed_caption(
        self,
        image_parts: list[dict],
        user_text: str = "",
        event_emitter: EventEmitter = None,
    ) -> Optional[str]:
        """Generate a rich, detailed description of images for the vision proxy.

        Unlike the short routing caption (1-2 sentences), this produces a thorough
        description that preserves all key details a text-only model would need to
        understand the image: visible text, error messages, code snippets, chart
        values, layout, colors, relationships.

        Returns the detailed caption string on success, or None on failure.
        """
        if not image_parts or not self.valves.IMAGE_CAPTION_MODEL:
            return None

        # Reuse already-resolved images from the routing caption call
        clean_parts = await self._resolve_image_urls(image_parts, event_emitter)
        if not clean_parts:
            return None

        # Rich, detailed prompt — the main model will rely on this to "see" the image
        prompt_text = (
            "Provide a thorough, detailed description of this image. "
            "This description will be read by a text-only AI that cannot see the image, "
            "so be as comprehensive as possible. Include:\n"
            "1. TYPE: What kind of image is this? (screenshot, photo, diagram, chart, "
            "document scan, whiteboard, meme, etc.)\n"
            "2. VISIBLE TEXT: Quote ALL visible text exactly as it appears — error messages, "
            "code, labels, titles, axis values, button text, filenames, URLs, numbers. "
            "Do NOT paraphrase or summarize text — reproduce it verbatim.\n"
            "3. LAYOUT & STRUCTURE: Describe the spatial arrangement — where things are "
            "positioned relative to each other, hierarchies, groupings, flows.\n"
            "4. COLORS & SHAPES: Key visual elements — color coding, icons, highlights, "
            "boxes, arrows, connections, annotations.\n"
            "5. CONTEXT: What is the overall subject? What is the user likely asking about? "
            "What action or problem does this image relate to?\n\n"
            "Be specific and precise. A developer reading this description should be able "
            "to understand and act on the image content without seeing it."
        )
        if user_text.strip():
            prompt_text += (
                f"\n\nThe user's message accompanying this image is: "
                f'"{user_text.strip()[:300]}"'
            )

        vision_content: list[dict] = [{"type": "text", "text": prompt_text}]
        vision_content.extend(clean_parts)

        # Call vision model with fallback chain
        caption = await self._call_vision_model(
            vision_content,
            self.valves.IMAGE_PROXY_MAX_TOKENS,
            event_emitter=event_emitter,
        )
        if caption is None:
            await self._emit_status(
                event_emitter,
                "⚠️ Detailed image caption failed (all models in fallback chain).",
            )
        return caption

    def _get_sticky(self, chat_id: Optional[str]) -> Optional[dict]:
        if not chat_id or not self.valves.ENABLE_STICKY_ROUTING:
            return None
        entry = self.sticky_routes.get(chat_id)
        if not entry:
            return None
        if time.time() - entry.get("timestamp", 0) > self.valves.STICKY_TTL_SECONDS:
            del self.sticky_routes[chat_id]
            return None
        self.sticky_routes.move_to_end(chat_id)
        return entry

    def _set_sticky(
        self, chat_id: Optional[str], category: str, searched: bool
    ) -> None:
        if not chat_id or not self.valves.ENABLE_STICKY_ROUTING:
            return
        self.sticky_routes[chat_id] = {
            "category": category,
            "searched": searched,
            "timestamp": time.time(),
        }
        self.sticky_routes.move_to_end(chat_id)
        while len(self.sticky_routes) > self.valves.STICKY_MAX_CONVOS:
            self.sticky_routes.popitem(last=False)

    # -----------------------------------------------------------------
    # Chat memory (Phase 1): persistent per-chat semantic recall.
    # Storage: SQLite at CHAT_MEMORY_DB_PATH inside OWUI's data volume.
    # Strictly chat-scoped — every query uses WHERE chat_id=?.
    # Fails open: any DB/embedding error logs a warning and memory
    # silently no-ops; the main reply path is never blocked.
    # -----------------------------------------------------------------

    def _get_memory_conn_lock(self) -> asyncio.Lock:
        if self._memory_conn_lock is None:
            self._memory_conn_lock = asyncio.Lock()
        return self._memory_conn_lock

    async def _get_memory_conn(self) -> Optional[sqlite3.Connection]:
        """Lazy-open the chat memory DB. None if disabled or init failed.

        The connection is shared across requests (check_same_thread=False
        plus SQLite's internal locking). WAL mode keeps reads non-blocking
        during writes. On first init failure we mark memory disabled for
        the life of the process to avoid repeated open-failure spam.
        """
        if not self.valves.ENABLE_CHAT_MEMORY or self._memory_disabled:
            return None
        if self._memory_conn is not None:
            return self._memory_conn
        async with self._get_memory_conn_lock():
            if self._memory_conn is not None:
                return self._memory_conn
            try:
                path = self.valves.CHAT_MEMORY_DB_PATH
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                conn = sqlite3.connect(path, check_same_thread=False, timeout=5)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        embedding BLOB,
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_hash "
                    "ON chat_turns(chat_id, content_hash)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat ON chat_turns(chat_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_created ON chat_turns(created_at)"
                )
                conn.commit()
                self._memory_conn = conn
                logger.info("Chat memory DB ready at %s", path)
                return conn
            except Exception as e:
                logger.warning(
                    "Chat memory DB unavailable (%s) — memory disabled for this process.",
                    e,
                )
                self._memory_disabled = True
                return None

    async def _store_chat_turn(
        self, chat_id: str, role: str, raw_content: str
    ) -> None:
        """Store one turn. Idempotent via (chat_id, content_hash).

        Strips mechanical detail first (think blocks, route tag, verification
        trailer) so stored text is just the actual human-visible content.
        Pure-acknowledgment turns ("ok", "thanks") are skipped — no retrieval
        value and they pollute top-K.
        """
        if not chat_id:
            return
        content = _clean_for_memory(raw_content)
        if not content or _ACK_ONLY_RE.match(content):
            return
        conn = await self._get_memory_conn()
        if conn is None:
            return
        try:
            ch = _memory_content_hash(content)
            # Cheap dedup check: avoid an embedding call if we already have it.
            already = conn.execute(
                "SELECT 1 FROM chat_turns WHERE chat_id=? AND content_hash=? LIMIT 1",
                (chat_id, ch),
            ).fetchone()
            if already:
                return
            vec = await self._get_embedding(content[:2000])
            emb_blob = _f32_pack(vec) if vec else None
            conn.execute(
                "INSERT OR IGNORE INTO chat_turns "
                "(chat_id, role, content, content_hash, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chat_id, role, content, ch, emb_blob, time.time()),
            )
            # Per-chat size cap — keep newest CHAT_MEMORY_MAX_TURNS_PER_CHAT.
            max_n = self.valves.CHAT_MEMORY_MAX_TURNS_PER_CHAT
            conn.execute(
                "DELETE FROM chat_turns WHERE id IN ("
                "  SELECT id FROM chat_turns WHERE chat_id=? "
                "  ORDER BY created_at DESC LIMIT -1 OFFSET ?"
                ")",
                (chat_id, max_n),
            )
            conn.commit()
        except Exception as e:
            logger.warning(
                "Chat memory store failed (chat=%s): %s", str(chat_id)[:20], e
            )

    async def _recall_chat_memories(
        self,
        chat_id: str,
        query: str,
        exclude_hashes: set,
    ) -> list[tuple[str, str]]:
        """Top-K most similar prior turns from THIS chat only.

        Returns list of (role, content). Empty when:
          - memory disabled / DB broken
          - chat has < CHAT_MEMORY_MIN_TURNS stored rows
          - query embedding fails
        `exclude_hashes` are content hashes already visible in the current
        body["messages"] — we don't re-inject what the model will already see.
        """
        if not chat_id or not query.strip():
            return []
        conn = await self._get_memory_conn()
        if conn is None:
            return []
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM chat_turns WHERE chat_id=?", (chat_id,)
            ).fetchone()[0]
            if total < self.valves.CHAT_MEMORY_MIN_TURNS:
                return []
            rows = list(
                conn.execute(
                    "SELECT role, content, content_hash, embedding "
                    "FROM chat_turns WHERE chat_id=? AND embedding IS NOT NULL",
                    (chat_id,),
                )
            )
            if not rows:
                return []
            qvec = await self._get_embedding(query[:2000])
            if not qvec:
                return []
            scored: list[tuple[float, str, str]] = []
            for role, content, ch, emb_blob in rows:
                if ch in exclude_hashes or not emb_blob:
                    continue
                try:
                    vec = _f32_unpack(emb_blob)
                except Exception:
                    continue
                score = _cosine_similarity(qvec, vec)
                scored.append((score, role, content))
            scored.sort(key=lambda t: t[0], reverse=True)
            top = scored[: self.valves.CHAT_MEMORY_TOP_K]
            return [(role, content) for _, role, content in top]
        except Exception as e:
            logger.warning(
                "Chat memory recall failed (chat=%s): %s", str(chat_id)[:20], e
            )
            return []

    async def _maybe_cleanup_memories(self) -> None:
        """Probabilistic GC (~1% of outlet calls). Two sweeps:

        1. TTL: delete rows older than CHAT_MEMORY_TTL_DAYS.
        2. Referential: delete rows for chat_ids that no longer exist in
           OWUI's webui.db 'chat' table (user deleted the chat in the UI).
        """
        if random.random() > 0.01:
            return
        conn = await self._get_memory_conn()
        if conn is None:
            return
        try:
            # 1. TTL sweep
            cutoff = time.time() - self.valves.CHAT_MEMORY_TTL_DAYS * 86400
            ttl_removed = conn.execute(
                "DELETE FROM chat_turns WHERE created_at < ?", (cutoff,)
            ).rowcount

            # 2. Referential sweep — cross-check against webui.db.chat.
            orphan_removed = 0
            webui_db = "/app/backend/data/webui.db"
            if os.path.exists(webui_db):
                try:
                    alive = sqlite3.connect(
                        f"file:{webui_db}?mode=ro", uri=True, timeout=2
                    )
                    try:
                        alive_ids = {r[0] for r in alive.execute("SELECT id FROM chat")}
                    finally:
                        alive.close()
                    if alive_ids:
                        memory_ids = {
                            r[0]
                            for r in conn.execute(
                                "SELECT DISTINCT chat_id FROM chat_turns"
                            )
                        }
                        orphans = memory_ids - alive_ids
                        if orphans:
                            qs = ",".join("?" * len(orphans))
                            orphan_removed = conn.execute(
                                f"DELETE FROM chat_turns WHERE chat_id IN ({qs})",
                                tuple(orphans),
                            ).rowcount
                except Exception as e:
                    logger.warning("Chat memory referential GC failed: %s", e)

            conn.commit()
            if ttl_removed or orphan_removed:
                logger.info(
                    "Chat memory GC: ttl_removed=%d orphan_removed=%d",
                    ttl_removed,
                    orphan_removed,
                )
        except Exception as e:
            logger.warning("Chat memory cleanup failed: %s", e)

    def _build_classifier_prompt(self, messages: list, image_caption: str = "") -> str:
        recent = messages[-3:]
        lines = []
        for m in recent:
            content = _extract_text(m.get("content")).strip()
            if not content:
                continue
            role = m.get("role", "user")
            if role == "assistant":
                content = _ROUTE_HEADER_RE.sub("", content, count=1).strip()
            lines.append(f"[{role}]: {_sanitize_query(content[:300])}")
        convo = "\n".join(lines) if lines else "[user]: (empty)"

        # Inject image caption context into the classifier prompt so it can
        # route based on what the user sent visually, not just the text.
        image_hint = ""
        if image_caption:
            image_hint = (
                f"\nNOTE: The user also sent an image. "
                f'Auto-generated description: "{image_caption}"\n'
            )

        return (
            "Classify the LAST user message into ONE category based on its topic and the prior context. "
            "Resolve pronouns like 'it'/'that' using the earlier turns.\n"
            "Categories: FACTUAL, REASONING, CODING, RESEARCH, CASUAL.\n\n"
            f"Recent messages:\n{convo}\n"
            f"{image_hint}\n"
            "Return ONLY the category name."
        )

    async def _llm_verify_citations(
        self, response: str, search_context: str
    ) -> tuple[bool, str]:
        prompt = (
            "Audit the RESPONSE against the SEARCH_RESULTS below.\n\n"
            "Rules:\n"
            "1. Every factual claim needs an inline citation. Accepted forms: [N] numbered refs, "
            "[Source: <url>], or a bare URL.\n"
            "2. If the response uses [N] numbered refs, it MUST end with a '**Sources:**' section "
            "mapping each N to a URL (or to 'Tavily AI Summary').\n"
            "3. Every cited URL (inline or in the Sources section) must appear in SEARCH_RESULTS. "
            "Invented URLs = hallucinations = FAIL.\n"
            "4. A ref mapping to 'Tavily AI Summary' is ALWAYS valid when SEARCH_RESULTS contains a "
            "'Tavily AI Summary' section — that section IS part of the results.\n"
            "5. Citations must plausibly support the claim they follow.\n\n"
            "CRITICAL OUTPUT FORMAT — exactly ONE line, nothing else:\n"
            "  PASS: <one-sentence reason>\n"
            "  FAIL: <one-sentence reason>\n\n"
            "Do NOT think step-by-step. Do NOT reason aloud. Do NOT restate the rules. "
            "Your ENTIRE output must be the single verdict line.\n\n"
            f"SEARCH_RESULTS:\n{search_context[:6000]}\n\n"
            f"RESPONSE:\n{response[:4000]}\n\n"
            "Verdict:"
        )
        verdict = await self._call_llm(
            prompt,
            self.valves.VERIFIER_MODEL,
            max_tokens=200,
            fallback_chain=VERIFIER_FALLBACK_CHAIN,
        )
        if not verdict:
            return True, "verifier LLM unavailable — fail-open"

        cleaned = _strip_thinking_blocks(verdict).strip()
        pass_match = re.search(r"\bPASS\b\s*:?\s*([^\n]*)", cleaned, re.IGNORECASE)
        fail_match = re.search(r"\bFAIL\b\s*:?\s*([^\n]*)", cleaned, re.IGNORECASE)

        if pass_match and (not fail_match or pass_match.start() < fail_match.start()):
            reason = pass_match.group(1).strip() or "citations verified"
            return True, reason
        if fail_match:
            reason = fail_match.group(1).strip() or "unspecified"
            return False, reason

        logger.warning(
            "Verifier verdict unparseable — fail-open. Raw: %s", cleaned[:200]
        )
        return True, "verdict unparseable — fail-open"

    async def _emit_status(
        self,
        event_emitter: EventEmitter,
        description: str,
        done: bool = True,
    ) -> None:
        if event_emitter is None or not self.valves.EMIT_STATUS_EVENTS:
            return
        try:
            await event_emitter(
                {
                    "type": "status",
                    "data": {"description": description, "done": done},
                }
            )
        except Exception as e:
            logger.warning("Event emit failed: %s", e)

    async def _emit_replace(
        self,
        event_emitter: EventEmitter,
        content: str,
    ) -> None:
        if event_emitter is None:
            return
        try:
            await event_emitter(
                {
                    "type": "replace",
                    "data": {"content": content},
                }
            )
        except Exception as e:
            logger.warning("Replace emit failed: %s", e)

    def _build_route_content(
        self,
        category: str,
        searched: bool,
        body_text: str,
        override_label: Optional[str] = None,
        override_emoji: Optional[str] = None,
        trailer: str = "",
    ) -> str:
        if not self.valves.SHOW_ROUTE_TAG:
            return body_text + trailer
        tag_emoji = {
            "FACTUAL": "🔍",
            "REASONING": "🧮",
            "CODING": "💻",
            "RESEARCH": "📚",
            "CASUAL": "💬",
        }
        emoji = override_emoji or tag_emoji.get(category, "💬")
        label = override_label or category
        header = f"`{emoji} {label}`\n"
        if searched:
            header += "> 🌐 **Tavily Search Executed**\n\n"
        else:
            header += "\n"
        return header + body_text + trailer

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: EventEmitter = None,
    ) -> dict:
        if not body.get("messages") or not self.valves.FIREWORKS_API_KEY:
            return body
        try:
            return await asyncio.wait_for(
                self._do_inlet(body, __user__, __event_emitter__),
                timeout=self.valves.INLET_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Inlet exceeded %ds — forwarding query unrouted.",
                self.valves.INLET_TIMEOUT,
            )
            await self._emit_status(
                __event_emitter__,
                f"⚠️ Router timed out after {self.valves.INLET_TIMEOUT}s — forwarding query unrouted.",
            )
            return body

    async def _do_inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: EventEmitter = None,
    ) -> dict:
        self._invalidate_stale_caches()

        messages = body["messages"]
        query_raw = _extract_text(messages[-1]["content"]).strip()
        query_lower = query_raw.lower()

        # --- Image-based routing augmentation ---
        # If the last user message contains images, generate a short caption
        # and inject it into the routing query. The original message (with
        # images) is NOT modified here — the vision proxy step later decides
        # whether to replace images based on the selected model.
        image_caption = ""
        image_parts: list[dict] = []  # populated if images are present
        last_content = messages[-1].get("content")
        if self.valves.ENABLE_IMAGE_ROUTING and isinstance(last_content, list):
            image_parts = _extract_images(last_content)
            if image_parts:
                await self._emit_status(
                    __event_emitter__,
                    f"🖼️ Router: captioning {len(image_parts)} image(s) for routing…",
                    done=False,
                )
                caption = await self._caption_images(
                    image_parts, user_text=query_raw, event_emitter=__event_emitter__
                )
                if caption:
                    image_caption = caption
                    await self._emit_status(
                        __event_emitter__,
                        "🖼️ Router: image captioned — routing enriched.",
                    )
                else:
                    await self._emit_status(
                        __event_emitter__,
                        "⚠️ Router: image captioning failed — falling back to text-only routing.",
                    )

        # Build the augmented routing query: user text + image context
        if image_caption:
            routing_query = f"{query_raw} [Image context: {image_caption}]".strip()
        else:
            routing_query = query_raw
        routing_query_lower = routing_query.lower()

        force_search = any(p.search(routing_query_lower) for p in FORCE_SEARCH_PATTERNS)
        chat_id = self._extract_chat_id(body)
        is_followup = _is_followup_query(query_raw)

        await self._ensure_anchor_embeddings()

        query_vec = await self._get_embedding(routing_query[:500])
        best_match = None
        highest_score = -1.0

        if query_vec:
            for category, vec in self.anchor_embeddings.items():
                score = _cosine_similarity(query_vec, vec)
                if score > highest_score:
                    highest_score, best_match = score, category

        sticky_searched = False
        if highest_score < self.valves.ROUTING_THRESHOLD:
            sticky = self._get_sticky(chat_id) if is_followup else None
            if sticky:
                best_match = sticky["category"]
                sticky_searched = bool(sticky.get("searched"))
                await self._emit_status(
                    __event_emitter__,
                    f"🧲 Router: inherited route ({best_match}) from previous turn.",
                )
            else:
                llm_response = await self._call_llm(
                    self._build_classifier_prompt(messages, image_caption),
                    self.valves.CLASSIFIER_MODEL,
                    fallback_chain=CLASSIFIER_FALLBACK_CHAIN,
                )
                if llm_response is None:
                    await self._emit_status(
                        __event_emitter__,
                        "⚠️ Router: classifier LLM unavailable — keeping embedding best-match.",
                    )
                llm_cat = self._parse_llm_category(llm_response)
                if llm_cat:
                    best_match = llm_cat

        if not best_match:
            await self._emit_status(
                __event_emitter__,
                "⚠️ Router: could not classify query — defaulting to CASUAL.",
            )
            best_match = "CASUAL"

        will_search = (
            best_match in ["FACTUAL", "RESEARCH"] or force_search or sticky_searched
        )
        search_flag = "_SEARCH" if will_search else ""
        self._set_sticky(chat_id, best_match, will_search)

        body.setdefault("metadata", {})
        body["metadata"]["_router_state"] = {
            "category": best_match,
            "searched": will_search,
        }

        # Triple fallback for routing state: metadata (primary) → sticky_routes (2nd) → tag (3rd).
        # Tag is the tertiary safety net — outlet may strip it from display if replace events work.
        system_content = (
            f"OUTPUT STRUCTURE:\n"
            f"1. If you want to reason, put it inside <think>...</think> tags (hidden from the user).\n"
            f"2. Then emit exactly this single line: [ROUTER_STATE: {best_match}{search_flag}]\n"
            f"3. Then write your answer for the user.\n"
            f"Do NOT narrate your plan or restate the question in visible output outside of a "
            f"<think>...</think> block.\n\n"
        )

        search_context = ""
        if will_search:
            depth = (
                self.valves.SEARCH_RESULTS_RESEARCH
                if best_match == "RESEARCH"
                else self.valves.SEARCH_RESULTS_FACTUAL
            )
            if sticky_searched and is_followup:
                prior_user_turns = [
                    _extract_text(m["content"])
                    for m in messages
                    if m.get("role") == "user" and m.get("content")
                ][-2:]
                search_query = " ".join(prior_user_turns)[:500] or routing_query
            else:
                search_query = routing_query
            await self._emit_status(
                __event_emitter__,
                f"🌐 Searching the web ({best_match}, depth={depth})…",
                done=False,
            )
            search_context = await self._search_tavily(search_query, max_results=depth)

            search_ok = (
                bool(search_context)
                and "[Web Search Failed" not in search_context
                and "[No Tavily API Key" not in search_context
            )
            if search_ok:
                system_content += (
                    f"=========================================\n"
                    f"{SEARCH_MARKER} (USE THESE OR FAIL):\n"
                    f"=========================================\n"
                    f"{search_context}\n\n"
                    f"=========================================\n\n"
                )
                body["metadata"]["_router_state"]["search_context"] = search_context
                await self._emit_status(
                    __event_emitter__,
                    f"✅ Web search complete ({best_match}).",
                )
            else:
                system_content += (
                    f"=========================================\n"
                    f"WEB SEARCH FAILED OR RETURNED NO DATA\n"
                    f"=========================================\n\n"
                )
                reason = (
                    "Tavily API key missing"
                    if "[No Tavily API Key" in search_context
                    else "upstream search failed"
                )
                await self._emit_status(
                    __event_emitter__,
                    f"⚠️ Web search unavailable ({reason}). Answering without live data.",
                )

        # --- Chat memory recall: inject relevant prior turns when the chat
        # is long enough to have meaningfully aged out of OWUI's visible
        # history. Strictly chat-scoped; skipped if no chat_id. ---
        if self.valves.ENABLE_CHAT_MEMORY and chat_id:
            exclude_hashes = set()
            for m in messages:
                cleaned = _clean_for_memory(_extract_text(m.get("content", "")))
                if cleaned:
                    exclude_hashes.add(_memory_content_hash(cleaned))
            recalled = await self._recall_chat_memories(
                chat_id, routing_query, exclude_hashes
            )
            if recalled:
                memory_block = (
                    "=========================================\n"
                    "RELEVANT PRIOR TURNS FROM THIS CHAT:\n"
                    "(semantic recall — use as additional context; each bracket is a prior turn)\n"
                    "=========================================\n"
                )
                for role, content in recalled:
                    snippet = content[:800].strip()
                    memory_block += f"[{role}] {snippet}\n\n"
                memory_block += "=========================================\n\n"
                system_content += memory_block
                await self._emit_status(
                    __event_emitter__,
                    f"🧠 Recalled {len(recalled)} prior turn(s) from this chat.",
                )

        system_content += (
            f"{self.prompts[best_match]}\n\n"
            f"ANTI-REFUSAL: Do NOT say 'I do not have internet access'. If search results were provided above, use them. "
            f"If search failed above, state that you cannot verify the live data."
        )

        existing_system = next(
            (i for i, m in enumerate(messages) if m.get("role") == "system"),
            None,
        )
        if existing_system is not None:
            sys_content = _extract_text(body["messages"][existing_system]["content"])
            body["messages"][existing_system]["content"] = (
                sys_content + f"\n\n{system_content}"
            )
        else:
            body["messages"].insert(0, {"role": "system", "content": system_content})

        # --- Vision proxy: replace images with detailed captions for non-vision models ---
        # If the selected model cannot natively process images, we replace each
        # image_url content part with a rich text description. This must run on
        # EVERY turn (not just when the current message has images) because
        # conversation history from previous turns still contains image_url parts
        # that the non-vision model will reject.
        selected_model = body.get("model", "")
        is_vision_model = selected_model in VISION_CAPABLE_MODELS

        if self.valves.ENABLE_VISION_PROXY and not is_vision_model:
            # Step 1: Scan ALL messages for image_url parts
            all_image_parts: list[tuple[int, dict]] = []  # (message_index, image_part)
            for i, m in enumerate(body["messages"]):
                content = m.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        all_image_parts.append((i, part))

            if all_image_parts:
                # Step 2: Resolve any internal URLs → base64 data URIs
                resolved = await self._resolve_image_urls(
                    [p for _, p in all_image_parts],
                    event_emitter=__event_emitter__,
                )

                # Step 3: Caption each unique image (with caching)
                # Use a cache keyed by chat_id + image URL to avoid re-captioning
                # the same image on follow-up turns.
                chat_id_for_cache = chat_id or "unknown"
                image_url_to_caption: dict[str, str] = {}

                # Group by URL to avoid duplicate caption calls
                unique_urls: dict[str, list[dict]] = {}  # url → list of resolved parts
                for part in resolved:
                    url = (part.get("image_url") or {}).get("url", "")
                    if url and url not in unique_urls:
                        unique_urls[url] = [part]

                new_captions = 0
                cached_captions = 0
                for url in unique_urls:
                    # SHA-256 the URL so that huge data: URIs and long internal
                    # paths can never collide via prefix match. Still chat-scoped.
                    url_digest = hashlib.sha256(url.encode()).hexdigest()[:32]
                    cache_key = f"{chat_id_for_cache}:{url_digest}"
                    if cache_key in self.image_caption_cache:
                        image_url_to_caption[url] = self.image_caption_cache[cache_key]
                        cached_captions += 1
                    else:
                        # Generate a new detailed caption
                        parts_for_url = unique_urls[url]
                        caption = await self._detailed_caption(
                            parts_for_url,
                            user_text=query_raw,
                            event_emitter=__event_emitter__,
                        )
                        if caption:
                            image_url_to_caption[url] = caption
                            self.image_caption_cache[cache_key] = caption
                            # Evict oldest entries if cache grows too large
                            while len(self.image_caption_cache) > 200:
                                self.image_caption_cache.popitem(last=False)
                            new_captions += 1
                        else:
                            image_url_to_caption[url] = (
                                "[Image description unavailable]"
                            )

                # Step 4: Replace image_url parts in ALL messages with captions
                total_replaced = 0
                for m in body["messages"]:
                    content = m.get("content")
                    if not isinstance(content, list):
                        continue
                    new_parts = []
                    has_images = False
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            url = (part.get("image_url") or {}).get("url", "")
                            caption_text = image_url_to_caption.get(
                                url, "[Attached image]"
                            )
                            new_parts.append(
                                {
                                    "type": "text",
                                    "text": f"\n[Attached image: {caption_text}]",
                                }
                            )
                            has_images = True
                            total_replaced += 1
                        else:
                            new_parts.append(part)
                    if has_images:
                        m["content"] = new_parts

                if new_captions > 0:
                    await self._emit_status(
                        __event_emitter__,
                        f"🖥️ Vision proxy: {selected_model.split('/')[-1]} doesn't support images — "
                        f"captioned {new_captions} new image(s), reused {cached_captions} cached, "
                        f"replaced {total_replaced} image part(s) in conversation.",
                    )
                elif cached_captions > 0:
                    await self._emit_status(
                        __event_emitter__,
                        f"🖥️ Vision proxy: reused {cached_captions} cached caption(s) — "
                        f"images replaced in conversation history.",
                    )
            elif image_parts:
                # Current message has images but they were already in routing image_parts
                # and none were found in body["messages"] as image_url parts
                # (shouldn't happen, but handle gracefully)
                pass

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: EventEmitter = None,
    ) -> dict:
        if not body.get("messages") or not self.valves.FIREWORKS_API_KEY:
            return body
        response = _extract_text(body["messages"][-1].get("content", ""))
        if not response:
            return body

        router_state = (body.get("metadata") or {}).get("_router_state") or {}
        category = router_state.get("category")
        search_results_injected = bool(router_state.get("searched"))
        search_context_from_inlet = router_state.get("search_context", "")

        # Fallback 1: Filter-instance sticky cache (survives even if OWUI strips body metadata).
        if not category:
            sticky_state = self._get_sticky(self._extract_chat_id(body))
            if sticky_state:
                category = sticky_state.get("category")
                search_results_injected = bool(sticky_state.get("searched"))

        # Fallback 2: parse the model's tag (only works if inlet still emits it).
        parse_copy = _strip_thinking_blocks(response)
        state_match = re.search(r"\[ROUTER_STATE:\s*([A-Z]+)(_SEARCH)?\]", parse_copy)
        if not category and state_match:
            category = state_match.group(1)
            search_results_injected = bool(state_match.group(2))

        if not category:
            category = "CASUAL"
            await self._emit_status(
                __event_emitter__,
                "⚠️ Router: no state available (metadata missing, model skipped tag) — rendering as CASUAL.",
            )

        display_response = _strip_thinking_blocks(response)
        tag_re = re.compile(r"\[ROUTER_STATE:\s*[A-Z]+(?:_SEARCH)?\]")
        tag_matches = list(tag_re.finditer(display_response))
        if tag_matches:
            last = tag_matches[-1]
            display_response = display_response[last.end() :].strip()
        else:
            display_response = display_response.strip()

        final_content = self._build_route_content(
            category, searched=search_results_injected, body_text=display_response
        )
        body["messages"][-1]["content"] = final_content
        await self._emit_replace(__event_emitter__, final_content)

        verifier_body = display_response

        if (
            self.valves.ENABLE_OUTLET_VERIFICATION
            and category in ["FACTUAL", "RESEARCH"]
            and len(verifier_body) >= 50
            and search_results_injected
        ):
            try:
                await asyncio.wait_for(
                    self._verify_and_correct(
                        body,
                        verifier_body,
                        category,
                        search_context_from_inlet,
                        __event_emitter__,
                    ),
                    timeout=self.valves.OUTLET_VERIFY_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Outlet verification exceeded %ds",
                    self.valves.OUTLET_VERIFY_TIMEOUT,
                )
                await self._emit_status(
                    __event_emitter__,
                    f"⚠️ Verifier timed out after {self.valves.OUTLET_VERIFY_TIMEOUT}s — stopping retries.",
                )

        # --- Chat memory: persist the finalized user+assistant turns and
        # run periodic GC. All operations are fail-open; a failure here
        # never reaches the user. ---
        if self.valves.ENABLE_CHAT_MEMORY:
            chat_id_out = self._extract_chat_id(body)
            if chat_id_out:
                user_msgs = [
                    m for m in body["messages"] if m.get("role") == "user"
                ]
                if user_msgs:
                    last_user_text = _extract_text(user_msgs[-1].get("content", ""))
                    await self._store_chat_turn(
                        chat_id_out, "user", last_user_text
                    )
                # Assistant content after all outlet massaging
                # (header wrap + optional verification trailer). The
                # _store_chat_turn path strips mechanical artifacts before
                # storing — the memory entry is just the actual answer.
                asst_text = _extract_text(body["messages"][-1].get("content", ""))
                await self._store_chat_turn(chat_id_out, "assistant", asst_text)
                await self._maybe_cleanup_memories()

        return body

    async def _verify_and_correct(
        self,
        body: dict,
        original_response: str,
        category: str,
        search_context: str,
        event_emitter: EventEmitter,
    ) -> None:
        if not search_context:
            search_context = next(
                (
                    _extract_text(m["content"])
                    for m in body["messages"]
                    if m.get("role") == "system"
                    and SEARCH_MARKER in _extract_text(m["content"])
                ),
                "",
            )

        mode = (self.valves.VERIFIER_MODE or "hybrid").lower()
        max_retries = max(0, self.valves.VERIFIER_MAX_RETRIES)

        user_msgs = [
            _extract_text(m["content"])
            for m in body["messages"]
            if m.get("role") == "user"
        ]
        user_query = user_msgs[-1] if user_msgs else "your previous prompt"
        category_prompt = self.prompts.get(category, "")
        safe_query = _sanitize_query(user_query)

        current_response = original_response

        for attempt in range(max_retries + 1):
            await self._emit_status(
                event_emitter,
                f"🔍 Verifying citations (attempt {attempt + 1}/{max_retries + 1})…",
                done=False,
            )

            passed = True
            reason = ""
            if mode in ("regex", "hybrid"):
                has_claims = bool(CLAIM_PATTERNS.search(current_response))
                has_citations = bool(CITATION_PATTERNS.search(current_response))
                if has_claims and not has_citations:
                    passed = False
                    reason = "regex: factual claims without inline citations"

            if passed and mode in ("llm", "hybrid"):
                passed, llm_reason = await self._llm_verify_citations(
                    current_response, search_context
                )
                reason = f"llm: {llm_reason}" if not passed else llm_reason

            if passed:
                if attempt == 0:
                    await self._emit_status(
                        event_emitter, f"✅ Verifier: {reason or 'citations verified.'}"
                    )
                else:
                    corrected_content = self._build_route_content(
                        category,
                        searched=True,
                        override_label=f"{category} → CITATION CORRECTED (attempt {attempt + 1})",
                        override_emoji="🔄",
                        body_text=current_response,
                    )
                    body["messages"][-1]["content"] = corrected_content
                    await self._emit_replace(event_emitter, corrected_content)
                    await self._emit_status(
                        event_emitter,
                        f"✅ Verifier: corrected on attempt {attempt + 1}.",
                    )
                return

            logger.info("[Verifier] attempt %d FAIL — %s", attempt + 1, reason)

            if attempt >= max_retries or not self.valves.VERIFIER_REGENERATE:
                preserved_content = self._build_route_content(
                    category,
                    searched=True,
                    body_text=original_response,
                    trailer=(
                        f"\n\n---\n"
                        f"⚠️ **Verification note:** Auto-verification couldn't confirm all "
                        f"citations after {attempt + 1} attempt(s). "
                        f"Last reason: {reason}. "
                        f"Double-check the linked sources before relying on the response above.\n"
                    ),
                )
                body["messages"][-1]["content"] = preserved_content
                await self._emit_replace(event_emitter, preserved_content)
                await self._emit_status(
                    event_emitter,
                    f"⚠️ Verifier: {attempt + 1} attempt(s) failed — original kept with warning.",
                )
                return

            await self._emit_status(
                event_emitter,
                f"🔄 Verifier FAIL ({reason}) — regenerating attempt {attempt + 2}…",
                done=False,
            )
            regen_prompt = (
                f"{category_prompt}\n\n"
                f"A previous attempt FAILED citation verification because: {reason}\n"
                f"Do better this time.\n\n"
                f"Answer the question using ONLY verified facts from the SEARCH_RESULTS below. "
                f"CITE SOURCES INLINE using numbered references [1], [2], etc. "
                f"Every number, date, percentage, name, or location must carry a [N] citation "
                f"whose URL appears verbatim in SEARCH_RESULTS. "
                f"At the END, add a '---\\n**Sources:**' section mapping each number to its URL. "
                f"Do NOT invent URLs.\n\n"
                f"Question: '{safe_query}'\n\n"
                f"SEARCH_RESULTS:\n{search_context[:6000]}"
            )
            new_response = await self._call_llm(
                regen_prompt,
                body.get("model") or self.valves.MAIN_MODEL,
                max_tokens=self.valves.VERIFIER_MAX_TOKENS,
            )
            if not new_response:
                await self._emit_status(
                    event_emitter,
                    "⚠️ Verifier: regeneration call failed — stopping retries.",
                )
                return

            if (
                "[DATA NOT FOUND]" in new_response
                or len(new_response) < len(original_response) * 0.4
            ):
                await self._emit_status(
                    event_emitter,
                    "⚠️ Verifier: regeneration degraded — keeping original with warning.",
                )
                preserved_content = self._build_route_content(
                    category,
                    searched=True,
                    body_text=original_response,
                    trailer=(
                        f"\n\n---\n"
                        f"⚠️ **Verification note:** Auto-verification couldn't confirm all citations "
                        f"({reason}). Double-check the linked sources before relying on the response above.\n"
                    ),
                )
                body["messages"][-1]["content"] = preserved_content
                await self._emit_replace(event_emitter, preserved_content)
                return

            current_response = new_response
