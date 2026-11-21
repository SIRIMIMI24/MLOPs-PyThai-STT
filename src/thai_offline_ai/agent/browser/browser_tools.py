"""Structured tool wrappers connecting BrowserSessionBackend to the JARVIS ToolRegistry."""

from __future__ import annotations

from typing import Any

from thai_offline_ai.agent.browser.browser_manager import BrowserObservation, BrowserSessionBackend
from thai_offline_ai.agent.permissions import sanitize_untrusted_web_content
from thai_offline_ai.agent.schemas import (
    BrowserClickArgs,
    BrowserCloseArgs,
    BrowserGetTabsArgs,
    BrowserOpenArgs,
    BrowserReadPageArgs,
    BrowserScrollArgs,
    BrowserSearchArgs,
    SourceLink,
    ToolResult,
)


def _obs_to_tool_result(
    tool_name: str, obs: BrowserObservation, *, max_chars: int = 4500
) -> ToolResult:
    sources: list[SourceLink] = []
    if obs.results:
        for item in obs.results:
            url = item.get("url", "").strip()
            if url.startswith(("http://", "https://")):
                sources.append(
                    SourceLink(
                        title=(item.get("title") or url)[:260],
                        url=url,
                        snippet=(item.get("snippet") or "")[:450],
                    )
                )
    elif obs.ok and obs.url.startswith(("http://", "https://")) and "duckduckgo.com" not in obs.url:
        sources.append(
            SourceLink(
                title=(obs.title or obs.url)[:260],
                url=obs.url,
                snippet=(obs.text[:220].replace("\n", " ").strip() if obs.text else ""),
            )
        )

    wrapped_text = ""
    injection_warning = False
    if obs.text:
        wrapped_text, injection_warning = sanitize_untrusted_web_content(
            obs.text, obs.url or tool_name, max_chars=max_chars
        )

    return ToolResult(
        tool_name=tool_name,
        ok=obs.ok,
        summary=obs.summary,
        data={
            "url": obs.url,
            "title": obs.title,
            "results": obs.results,
            "tabs": obs.tabs,
        },
        sources=sources,
        untrusted_content=wrapped_text,
        blocked_reason=obs.blocked_reason,
        prompt_injection_warning=injection_warning,
    )


def register_browser_tools(registry: Any, browser: BrowserSessionBackend, *, max_page_chars: int = 4500) -> None:
    """Register the 7 Phase 8.1 browser automation tools into ToolRegistry."""

    def handle_open(args: BrowserOpenArgs) -> ToolResult:
        obs = browser.open_url(args.url, new_tab=args.new_tab)
        return _obs_to_tool_result("browser_open", obs, max_chars=max_page_chars)

    def handle_search(args: BrowserSearchArgs) -> ToolResult:
        obs = browser.search(args.query, max_results=args.max_results)
        return _obs_to_tool_result("browser_search", obs, max_chars=max_page_chars)

    def handle_read(args: BrowserReadPageArgs) -> ToolResult:
        limit = min(args.max_chars, max_page_chars)
        obs = browser.read_page(max_chars=limit, tab_index=args.tab_index)
        return _obs_to_tool_result("browser_read_page", obs, max_chars=limit)

    def handle_click(args: BrowserClickArgs) -> ToolResult:
        obs = browser.click(args.selector_or_text)
        return _obs_to_tool_result("browser_click", obs, max_chars=max_page_chars)

    def handle_scroll(args: BrowserScrollArgs) -> ToolResult:
        obs = browser.scroll(direction=args.direction, amount=args.amount)
        return _obs_to_tool_result("browser_scroll", obs, max_chars=max_page_chars)

    def handle_tabs(_args: BrowserGetTabsArgs) -> ToolResult:
        obs = browser.get_tabs()
        return _obs_to_tool_result("browser_get_tabs", obs, max_chars=max_page_chars)

    def handle_close(_args: BrowserCloseArgs) -> ToolResult:
        obs = browser.close()
        return _obs_to_tool_result("browser_close", obs, max_chars=max_page_chars)

    registry.register(
        name="browser_search",
        description="Search public web pages using DuckDuckGo in the visible browser and return top clickable links + snippets.",
        args_model=BrowserSearchArgs,
        handler=handle_search,
        requires_online=True,
    )
    registry.register(
        name="browser_open",
        description="Open a public http(s) URL in the visible browser and extract its title and readable text.",
        args_model=BrowserOpenArgs,
        handler=handle_open,
        requires_online=True,
    )
    registry.register(
        name="browser_read_page",
        description="Read visible text from the active browser tab (or switch by tab_index).",
        args_model=BrowserReadPageArgs,
        handler=handle_read,
        requires_online=True,
    )
    registry.register(
        name="browser_click",
        description="Click a visible link or button in the active browser tab (requires user approval).",
        args_model=BrowserClickArgs,
        handler=handle_click,
        requires_online=True,
    )
    registry.register(
        name="browser_scroll",
        description="Scroll the active browser page up, down, to top, or to bottom.",
        args_model=BrowserScrollArgs,
        handler=handle_scroll,
        requires_online=True,
    )
    registry.register(
        name="browser_get_tabs",
        description="List currently open tabs controlled by the JARVIS browser agent.",
        args_model=BrowserGetTabsArgs,
        handler=handle_tabs,
        requires_online=False,
    )
    registry.register(
        name="browser_close",
        description="Close the JARVIS visible browser session.",
        args_model=BrowserCloseArgs,
        handler=handle_close,
        requires_online=False,
    )
