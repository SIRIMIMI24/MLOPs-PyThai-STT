"""Phase 8.1 — Playwright Python async browser manager running visible Chromium."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import re
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

from thai_offline_ai.agent.config import BrowserAgentConfig
from thai_offline_ai.errors import WorkbenchError

_CAPTCHA_PATTERNS = re.compile(
    r"(recaptcha|hcaptcha|cf-turnstile|cloudflare.*verify|verify\s+you\s+are\s+human|"
    r"are\s+you\s+a\s+robot|unusual\s+traffic|complete\s+the\s+security\s+check|"
    r"ยืนยันว่าคุณไม่ใช่โปรแกรมอัตโนมัติ)",
    re.IGNORECASE,
)

_AUTH_WALL_PATTERNS = re.compile(
    r"(sign\s+in\s+to\s+continue|log\s+in\s+to\s+your\s+account|enter\s+your\s+password|"
    r"เข้าสู่ระบบเพื่อดำเนินการต่อ)",
    re.IGNORECASE,
)


def playwright_is_installed() -> bool:
    """Check whether the optional playwright package is discoverable without importing it."""
    return importlib.util.find_spec("playwright") is not None


@dataclass(slots=True)
class BrowserObservation:
    ok: bool
    summary: str
    url: str = ""
    title: str = ""
    text: str = ""
    results: list[dict[str, str]] = field(default_factory=list)
    tabs: list[dict[str, Any]] = field(default_factory=list)
    blocked_reason: str | None = None


class BrowserSessionBackend(Protocol):
    """Protocol implemented by PlaywrightBrowserManager and deterministic test fakes."""

    @property
    def is_running(self) -> bool: ...

    @property
    def current_url(self) -> str | None: ...

    def open_url(self, url: str, *, new_tab: bool = False) -> BrowserObservation: ...

    def search(self, query: str, *, max_results: int = 5) -> BrowserObservation: ...

    def read_page(
        self, *, max_chars: int = 4000, tab_index: int | None = None
    ) -> BrowserObservation: ...

    def click(self, selector_or_text: str) -> BrowserObservation: ...

    def scroll(self, *, direction: str = "down", amount: int = 700) -> BrowserObservation: ...

    def get_tabs(self) -> BrowserObservation: ...

    def close(self) -> BrowserObservation: ...


def _unwrap_search_url(href: str) -> str:
    """Resolve DuckDuckGo redirect links (/l/?uddg=...) to clean public URLs."""
    raw = (href or "").strip()
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urlparse(raw)
    if "duckduckgo.com" in (parsed.netloc or "") and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        target = qs.get("uddg", [""])[0]
        if target:
            return unquote(target)
    return raw


class PlaywrightBrowserManager:
    """Manages one dedicated headed Chromium profile via Playwright's async API."""

    def __init__(self, config: BrowserAgentConfig) -> None:
        self.config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._playwright: Any = None
        self._context: Any = None
        self._active_page: Any = None
        self._last_url: str | None = None

    @property
    def is_running(self) -> bool:
        return self._context is not None

    @property
    def current_url(self) -> str | None:
        return self._last_url

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            ready = threading.Event()
            holder: dict[str, asyncio.AbstractEventLoop] = {}

            def runner() -> None:
                if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
                    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                holder["loop"] = loop
                ready.set()
                loop.run_forever()

            self._thread = threading.Thread(
                target=runner, name="jarvis-playwright-loop", daemon=True
            )
            self._thread.start()
            ready.wait(timeout=5.0)
            self._loop = holder["loop"]
            return self._loop

    def _run_coro(self, coro: Any, timeout: float = 35.0) -> Any:
        loop = self._ensure_loop()
        future: Future[Any] = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    async def _ensure_browser_async(self) -> Any:
        if (
            self._context is not None
            and self._active_page is not None
            and not self._active_page.is_closed()
        ):
            return self._active_page
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise WorkbenchError(
                "Playwright is not installed. Run: python scripts/setup_playwright.py"
            ) from exc

        profile_path = Path(self.config.profile_dir)
        profile_path.mkdir(parents=True, exist_ok=True)
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        if self._context is None:
            try:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(profile_path),
                    headless=not self.config.headed,
                    viewport={
                        "width": self.config.viewport_width,
                        "height": self.config.viewport_height,
                    },
                    args=["--disable-blink-features=AutomationControlled"],
                )
            except Exception as exc:
                raise WorkbenchError(
                    "Could not start Playwright Chromium. Run: python scripts/setup_playwright.py "
                    f"({type(exc).__name__}: {exc})"
                ) from exc
            self._context.set_default_navigation_timeout(self.config.navigation_timeout_ms)
            self._context.set_default_timeout(self.config.action_timeout_ms)

        pages = [p for p in self._context.pages if not p.is_closed()]
        self._active_page = pages[0] if pages else await self._context.new_page()
        return self._active_page

    async def _inspect_page_state(self, page: Any, max_chars: int) -> BrowserObservation:
        url = page.url or ""
        self._last_url = url
        try:
            title = (await page.title()) or url
        except Exception:  # noqa: BLE001
            title = url
        try:
            body_text = await page.evaluate(
                """() => {
                    const clone = document.body ? document.body.cloneNode(true) : null;
                    if (!clone) return '';
                    for (const tag of clone.querySelectorAll('script, style, noscript, svg')) {
                        tag.remove();
                    }
                    return (clone.innerText || '').replace(/\\s+\\n/g, '\\n').trim();
                }"""
            )
        except Exception:  # noqa: BLE001
            body_text = ""

        snippet = str(body_text or "")[:max_chars]
        if _CAPTCHA_PATTERNS.search(snippet[:2000]) or _CAPTCHA_PATTERNS.search(title):
            return BrowserObservation(
                ok=False,
                summary=(
                    f"Stopped at '{title}' because the website displayed a CAPTCHA or bot-verification gate. "
                    "JARVIS will not bypass CAPTCHAs; please complete it manually in the visible browser if desired."
                ),
                url=url,
                title=title,
                text="",
                blocked_reason="captcha_or_bot_gate",
            )
        if _AUTH_WALL_PATTERNS.search(snippet[:1500]) and len(snippet) < 800:
            return BrowserObservation(
                ok=False,
                summary=(
                    f"Page '{title}' requires authentication. JARVIS does not bypass login walls."
                ),
                url=url,
                title=title,
                text=snippet,
                blocked_reason="authentication_required",
            )
        return BrowserObservation(
            ok=True,
            summary=f"Loaded '{title}' ({url}).",
            url=url,
            title=title,
            text=snippet,
        )

    async def _dismiss_common_consent(self, page: Any) -> None:
        """Best-effort dismissal of simple cookie consent banners without touching forms."""
        for label in ("Accept all", "Reject all", "I agree", "ยอมรับทั้งหมด", "ปิด"):
            with contextlib.suppress(Exception):
                btn = page.get_by_role("button", name=label, exact=True).first
                if await btn.is_visible(timeout=400):
                    await btn.click(timeout=1000)
                    return

    async def _open_url_async(self, url: str, *, new_tab: bool) -> BrowserObservation:
        page = await self._ensure_browser_async()
        if new_tab:
            open_pages = [p for p in self._context.pages if not p.is_closed()]
            if len(open_pages) >= self.config.max_tabs:
                await open_pages[0].close()
            page = await self._context.new_page()
            self._active_page = page
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=self.config.navigation_timeout_ms)
            await page.wait_for_timeout(600)
            await self._dismiss_common_consent(page)
        except Exception as exc:  # noqa: BLE001
            return BrowserObservation(
                ok=False,
                summary=f"Could not open {url}: {type(exc).__name__} ({exc})",
                url=url,
                blocked_reason="navigation_error",
            )
        return await self._inspect_page_state(page, self.config.max_page_chars)

    async def _search_async(self, query: str, *, max_results: int) -> BrowserObservation:
        page = await self._ensure_browser_async()
        search_url = f"https://duckduckgo.com/?q={quote_plus(query)}&ia=web"
        try:
            await page.goto(
                search_url, wait_until="domcontentloaded", timeout=self.config.navigation_timeout_ms
            )
            await page.wait_for_timeout(1200)
        except Exception as exc:  # noqa: BLE001
            return BrowserObservation(
                ok=False,
                summary=(
                    f"Web search failed because the search engine could not be reached "
                    f"({type(exc).__name__}). Check your internet connection."
                ),
                url=search_url,
                blocked_reason="network_unreachable",
            )

        state = await self._inspect_page_state(page, max_chars=2500)
        if state.blocked_reason:
            return state

        raw_results: list[dict[str, str]] = []
        try:
            extracted = await page.evaluate(
                """(limit) => {
                    const out = [];
                    const seen = new Set();
                    const cards = document.querySelectorAll(
                        'article[data-testid="result"], .result, .results_links'
                    );
                    for (const card of cards) {
                        const link = card.querySelector(
                            'a[data-testid="result-title-a"], a.result__a, h2 a'
                        );
                        if (!link || !link.href) continue;
                        const href = link.href;
                        if (seen.has(href)) continue;
                        seen.add(href);
                        const snippetEl = card.querySelector(
                            '[data-result="snippet"], .result__snippet'
                        );
                        out.push({
                            title: (link.innerText || href).trim(),
                            url: href,
                            snippet: snippetEl ? (snippetEl.innerText || '').trim() : ''
                        });
                        if (out.length >= limit) break;
                    }
                    return out;
                }""",
                max_results,
            )
            for item in extracted or []:
                clean_url = _unwrap_search_url(str(item.get("url", "")))
                if clean_url.startswith(("http://", "https://")):
                    raw_results.append(
                        {
                            "title": str(item.get("title", clean_url))[:260],
                            "url": clean_url,
                            "snippet": str(item.get("snippet", ""))[:450],
                        }
                    )
        except Exception:  # noqa: BLE001
            raw_results = []

        # Fallback to DuckDuckGo HTML endpoint if JS selectors returned nothing.
        if not raw_results:
            html_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            with contextlib.suppress(Exception):
                await page.goto(
                    html_url, wait_until="domcontentloaded", timeout=self.config.navigation_timeout_ms
                )
                await page.wait_for_timeout(500)
                extracted_html = await page.evaluate(
                    """(limit) => {
                        const out = [];
                        for (const card of document.querySelectorAll('.result')) {
                            const link = card.querySelector('a.result__a');
                            if (!link || !link.href) continue;
                            const snip = card.querySelector('.result__snippet');
                            out.push({
                                title: (link.innerText || link.href).trim(),
                                url: link.href,
                                snippet: snip ? (snip.innerText || '').trim() : ''
                            });
                            if (out.length >= limit) break;
                        }
                        return out;
                    }""",
                    max_results,
                )
                for item in extracted_html or []:
                    clean_url = _unwrap_search_url(str(item.get("url", "")))
                    if clean_url.startswith(("http://", "https://")):
                        raw_results.append(
                            {
                                "title": str(item.get("title", clean_url))[:260],
                                "url": clean_url,
                                "snippet": str(item.get("snippet", ""))[:450],
                            }
                        )

        self._last_url = page.url or search_url
        if not raw_results:
            return BrowserObservation(
                ok=True,
                summary=f"Searched '{query}' on DuckDuckGo but extracted 0 structured links.",
                url=self._last_url,
                title=f"Search: {query}",
                text=state.text,
                results=[],
            )
        lines = [
            f"{idx}. {item['title']} — {item['url']}\n   {item['snippet']}"
            for idx, item in enumerate(raw_results, start=1)
        ]
        return BrowserObservation(
            ok=True,
            summary=f"Found {len(raw_results)} web search results for '{query}'.",
            url=self._last_url,
            title=f"Search: {query}",
            text="\n".join(lines),
            results=raw_results,
        )

    async def _read_page_async(
        self, *, max_chars: int, tab_index: int | None
    ) -> BrowserObservation:
        if self._context is None:
            return BrowserObservation(
                ok=False,
                summary="No active browser session. Open or search a page first.",
                blocked_reason="browser_not_started",
            )
        pages = [p for p in self._context.pages if not p.is_closed()]
        if not pages:
            return BrowserObservation(
                ok=False,
                summary="Browser has no open tabs.",
                blocked_reason="no_open_tabs",
            )
        if tab_index is not None:
            if not 0 <= tab_index < len(pages):
                return BrowserObservation(
                    ok=False,
                    summary=f"Tab index {tab_index} out of range (0..{len(pages) - 1}).",
                    blocked_reason="invalid_tab",
                )
            self._active_page = pages[tab_index]
            await self._active_page.bring_to_front()
        elif self._active_page is None or self._active_page.is_closed():
            self._active_page = pages[-1]
        return await self._inspect_page_state(self._active_page, max_chars=max_chars)

    async def _click_async(self, selector_or_text: str) -> BrowserObservation:
        if self._active_page is None or self._active_page.is_closed():
            return BrowserObservation(
                ok=False,
                summary="No active webpage to click.",
                blocked_reason="browser_not_started",
            )
        page = self._active_page
        try:
            locator = page.get_by_text(selector_or_text, exact=False).first
            if await locator.is_visible(timeout=1500):
                await locator.click(timeout=self.config.action_timeout_ms)
            else:
                await page.locator(selector_or_text).first.click(
                    timeout=self.config.action_timeout_ms
                )
            await page.wait_for_timeout(700)
        except Exception as exc:  # noqa: BLE001
            return BrowserObservation(
                ok=False,
                summary=f"Could not click '{selector_or_text}': {type(exc).__name__}",
                url=page.url or "",
                blocked_reason="click_failed",
            )
        return await self._inspect_page_state(page, max_chars=self.config.max_page_chars)

    async def _scroll_async(self, *, direction: str, amount: int) -> BrowserObservation:
        if self._active_page is None or self._active_page.is_closed():
            return BrowserObservation(
                ok=False,
                summary="No active webpage to scroll.",
                blocked_reason="browser_not_started",
            )
        page = self._active_page
        try:
            if direction == "top":
                await page.evaluate("window.scrollTo(0, 0)")
            elif direction == "bottom":
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            elif direction == "up":
                await page.evaluate(f"window.scrollBy(0, -{int(amount)})")
            else:
                await page.evaluate(f"window.scrollBy(0, {int(amount)})")
            await page.wait_for_timeout(300)
        except Exception as exc:  # noqa: BLE001
            return BrowserObservation(
                ok=False,
                summary=f"Scroll failed: {type(exc).__name__}",
                url=page.url or "",
            )
        obs = await self._inspect_page_state(page, max_chars=min(2000, self.config.max_page_chars))
        obs.summary = f"Scrolled {direction} on '{obs.title}'."
        return obs

    async def _get_tabs_async(self) -> BrowserObservation:
        if self._context is None:
            return BrowserObservation(ok=True, summary="No active browser tabs.", tabs=[])
        tabs: list[dict[str, Any]] = []
        for idx, page in enumerate(p for p in self._context.pages if not p.is_closed()):
            try:
                title = await page.title()
            except Exception:  # noqa: BLE001
                title = page.url or "Untitled"
            tabs.append(
                {
                    "index": idx,
                    "title": title,
                    "url": page.url or "",
                    "active": page == self._active_page,
                }
            )
        return BrowserObservation(
            ok=True,
            summary=f"{len(tabs)} agent-controlled browser tab(s) open.",
            url=self._last_url or "",
            tabs=tabs,
        )

    async def _close_async(self) -> BrowserObservation:
        try:
            if self._context is not None:
                await self._context.close()
        finally:
            self._context = None
            self._active_page = None
            self._last_url = None
            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                finally:
                    self._playwright = None
        return BrowserObservation(ok=True, summary="Closed agent browser session.")

    # Synchronous bridge methods used by ToolExecutor / Gradio / CLI
    def open_url(self, url: str, *, new_tab: bool = False) -> BrowserObservation:
        return self._run_coro(self._open_url_async(url, new_tab=new_tab))

    def search(self, query: str, *, max_results: int = 5) -> BrowserObservation:
        return self._run_coro(self._search_async(query, max_results=max_results))

    def read_page(
        self, *, max_chars: int = 4000, tab_index: int | None = None
    ) -> BrowserObservation:
        return self._run_coro(self._read_page_async(max_chars=max_chars, tab_index=tab_index))

    def click(self, selector_or_text: str) -> BrowserObservation:
        return self._run_coro(self._click_async(selector_or_text))

    def scroll(self, *, direction: str = "down", amount: int = 700) -> BrowserObservation:
        return self._run_coro(self._scroll_async(direction=direction, amount=amount))

    def get_tabs(self) -> BrowserObservation:
        return self._run_coro(self._get_tabs_async())

    def close(self) -> BrowserObservation:
        if self._loop is None or not self._loop.is_running():
            self._context = None
            self._active_page = None
            self._last_url = None
            return BrowserObservation(ok=True, summary="Browser session was not running.")
        return self._run_coro(self._close_async())
