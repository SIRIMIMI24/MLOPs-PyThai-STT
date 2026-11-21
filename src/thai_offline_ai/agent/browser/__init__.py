"""Phase 8.1 — Local Visible Browser Agent subpackage."""

from __future__ import annotations

from thai_offline_ai.agent.browser.browser_manager import (
    BrowserObservation,
    BrowserSessionBackend,
    PlaywrightBrowserManager,
    playwright_is_installed,
)
from thai_offline_ai.agent.browser.browser_tools import register_browser_tools

__all__ = [
    "BrowserObservation",
    "BrowserSessionBackend",
    "PlaywrightBrowserManager",
    "playwright_is_installed",
    "register_browser_tools",
]
