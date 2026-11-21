"""Modern SaaS design system, inline CSS, and pure-offline HTML/JS components.

Inspired by clean modern marketplace/SaaS UIs (high-contrast typography, generous
whitespace, cobalt blue accents, structured cards) with an original identity for
Thai Offline AI Workbench. Zero external fonts, CDNs, or network requests.
"""
from __future__ import annotations

import os
import platform
from collections.abc import Sequence
from html import escape
from typing import Any

from thai_offline_ai.config import Settings

SYSTEM_SANS_FONTS = [
    "Noto Sans Thai",
    "Inter",
    "Segoe UI",
    "Leelawadee UI",
    "Tahoma",
    "system-ui",
    "sans-serif",
]
SYSTEM_MONO_FONTS = [
    "Cascadia Code",
    "JetBrains Mono",
    "Consolas",
    "ui-monospace",
    "monospace",
]

HEAD_SCRIPT = """<script>
(function () {
  function enforceLightMode() {
    if (document.documentElement) document.documentElement.classList.remove("dark");
    if (document.body) document.body.classList.remove("dark");
  }

  enforceLightMode();
  document.addEventListener("DOMContentLoaded", function () {
    enforceLightMode();
    if (typeof MutationObserver !== "undefined" && document.body) {
      var observer = new MutationObserver(function () {
        if (document.body.classList.contains("dark") || document.documentElement.classList.contains("dark")) {
          enforceLightMode();
        }
      });
      observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
      observer.observe(document.body, { attributes: true, attributeFilter: ["class"] });
    }
  });

  function activateTargetTab(el) {
    var target = (el.getAttribute("data-target-tab") || "").trim();
    if (!target) return;
    var buttons = Array.prototype.slice.call(
      document.querySelectorAll(".tab-nav button, [role='tab']")
    );
    var match = buttons.find(function (btn) {
      return (btn.textContent || "").trim() === target;
    });
    if (!match) return;
    match.click();

    var accordionHint = (el.getAttribute("data-open-accordion") || "").trim();
    if (accordionHint) {
      window.setTimeout(function () {
        var accButtons = Array.prototype.slice.call(
          document.querySelectorAll(".accordion button.label-wrap, button.label-wrap")
        );
        var acc = accButtons.find(function (btn) {
          return (btn.textContent || "").indexOf(accordionHint) !== -1;
        });
        if (acc && acc.getAttribute("aria-expanded") !== "true") {
          acc.click();
        }
      }, 140);
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  document.addEventListener("click", function (event) {
    var node = event.target && event.target.closest ? event.target.closest("[data-target-tab]") : null;
    if (node) activateTargetTab(node);
  });

  document.addEventListener("keydown", function (event) {
    if (event.key !== "Enter" && event.key !== " ") return;
    var node = event.target && event.target.closest ? event.target.closest("[data-target-tab]") : null;
    if (node) {
      event.preventDefault();
      activateTargetTab(node);
    }
  });
})();
</script>"""

CUSTOM_CSS = """
:root,
html,
body,
.dark,
.gradio-container {
    color-scheme: light !important;
    --primary: #2563EB;
    --primary-hover: #1D4ED8;
    --primary-deep: #1E3A8A;
    --primary-soft: #EFF6FF;
    --primary-border: #BFDBFE;
    --bg: #F8FAFC;
    --surface: #FFFFFF;
    --surface-subtle: #F1F5F9;
    --text-primary: #0F172A;
    --text-secondary: #64748B;
    --border: #E2E8F0;
    --border-strong: #CBD5E1;
    --success: #10B981;
    --success-text: #047857;
    --success-soft: #ECFDF5;
    --success-border: #A7F3D0;
    --warn-text: #B45309;
    --warn-soft: #FFFBEB;
    --warn-border: #FDE68A;
    --err-text: #B91C1C;
    --err-soft: #FEF2F2;
    --err-border: #FECACA;
    --radius-sm: 10px;
    --radius-md: 14px;
    --radius-lg: 18px;
    --shadow-xs: 0 1px 2px rgba(15, 23, 42, 0.04);
    --shadow-sm: 0 4px 16px rgba(15, 23, 42, 0.045);
    --shadow-md: 0 10px 28px rgba(37, 99, 235, 0.10);

    /* Pin Gradio internal variables to light palette */
    --body-background-fill: #F8FAFC !important;
    --body-text-color: #0F172A !important;
    --body-text-color-subdued: #64748B !important;
    --background-fill-primary: #FFFFFF !important;
    --background-fill-secondary: #F8FAFC !important;
    --block-background-fill: #FFFFFF !important;
    --block-border-color: #E2E8F0 !important;
    --block-label-background-fill: #FFFFFF !important;
    --block-label-text-color: #0F172A !important;
    --block-title-text-color: #0F172A !important;
    --block-info-text-color: #64748B !important;
    --input-background-fill: #FFFFFF !important;
    --input-border-color: #CBD5E1 !important;
    --input-placeholder-color: #94A3B8 !important;
    --checkbox-label-text-color: #0F172A !important;
    --checkbox-label-background-fill: #F8FAFC !important;
    --checkbox-label-background-fill-selected: #EFF6FF !important;
    --checkbox-label-text-color-selected: #1D4ED8 !important;
    --accordion-text-color: #0F172A !important;
    --table-text-color: #0F172A !important;
    --table-even-background-fill: #FFFFFF !important;
    --table-odd-background-fill: #F8FAFC !important;
    --code-background-fill: #EFF6FF !important;
    --neutral-800: #1E293B !important;
    --neutral-900: #0F172A !important;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text-primary) !important;
    font-family: "Noto Sans Thai", "Inter", "Segoe UI", "Leelawadee UI", "Tahoma", system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* Adaptive Desktop-First Container (1380px) */
.gradio-container {
    max-width: 1380px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 18px 28px 36px 28px !important;
    box-sizing: border-box !important;
}

/* Hide Gradio default footer ("Use via API · Built with Gradio · Settings") */
footer,
.built-with,
.show-api,
a[href*="gradio.app"] {
    display: none !important;
}

/* High-contrast text guarantee across all Gradio widgets */
.gradio-container label,
.gradio-container label span,
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong,
.gradio-container .accordion button,
.gradio-container .accordion button span,
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container select,
.gradio-container .wrap-inner,
.gradio-container .secondary-wrap {
    color: var(--text-primary) !important;
}

.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container select {
    background: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus,
.gradio-container input[type="number"]:focus,
.gradio-container select:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.14) !important;
    outline: none !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: #94A3B8 !important;
    opacity: 1 !important;
}

.gradio-container code {
    background: var(--primary-soft) !important;
    color: var(--primary-hover) !important;
    padding: 2px 7px !important;
    border-radius: 6px !important;
    font-size: 0.88em !important;
    font-family: "Cascadia Code", "Consolas", monospace !important;
}

/* Top SaaS Application Header */
.aiw-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 14px 22px;
    margin-bottom: 14px;
    box-shadow: var(--shadow-sm);
}
.aiw-brand {
    display: flex;
    align-items: center;
    gap: 13px;
    cursor: pointer;
    border-radius: 10px;
}
.aiw-brand:focus-visible {
    outline: 2px solid var(--primary);
    outline-offset: 4px;
}
.aiw-logo {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 65%, #3B82F6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 6px 14px rgba(37, 99, 235, 0.25);
    flex-shrink: 0;
}
.aiw-brand-name {
    font-size: 18px;
    font-weight: 800;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.aiw-brand-name span {
    color: var(--primary) !important;
}
.aiw-brand-tagline {
    font-size: 12.5px;
    color: var(--text-secondary) !important;
    margin-top: 2px;
}
.aiw-header-status {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}

/* Reusable Badges / Pills */
.aiw-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text-secondary) !important;
    white-space: nowrap;
}
.aiw-pill-blue {
    border-color: var(--primary-border);
    background: var(--primary-soft);
    color: var(--primary-hover) !important;
}
.aiw-pill-success {
    border-color: var(--success-border);
    background: var(--success-soft);
    color: var(--success-text) !important;
}
.aiw-pill-warn {
    border-color: var(--warn-border);
    background: var(--warn-soft);
    color: var(--warn-text) !important;
}
.aiw-pill-err {
    border-color: var(--err-border);
    background: var(--err-soft);
    color: var(--err-text) !important;
}
.aiw-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--success);
    display: inline-block;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.22);
}

/* Primary Horizontal Navigation (Gradio Tab Bar) */
.tabs > .tab-nav {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 6px 8px !important;
    gap: 6px !important;
    margin-bottom: 22px !important;
    box-shadow: var(--shadow-xs) !important;
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
}
.tabs > .tab-nav > button {
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 9px 18px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    white-space: nowrap !important;
    transition: background 0.15s ease, color 0.15s ease, box-shadow 0.15s ease !important;
}
.tabs > .tab-nav > button:hover {
    background: var(--primary-soft) !important;
    color: var(--primary) !important;
}
.tabs > .tab-nav > button:focus-visible {
    outline: 2px solid var(--primary) !important;
    outline-offset: -2px !important;
}
.tabs > .tab-nav > button.selected {
    background: var(--primary) !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 10px rgba(37, 99, 235, 0.28) !important;
}

/* Page Header inside each Studio Tab */
.aiw-page-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: var(--shadow-xs);
}
.aiw-page-title {
    margin: 0 !important;
    font-size: 22px !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
    line-height: 1.25 !important;
}
.aiw-page-sub {
    margin: 5px 0 0 0 !important;
    font-size: 14px !important;
    color: var(--text-secondary) !important;
    line-height: 1.5 !important;
}
.aiw-page-badges {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}

/* Home Hero */
.aiw-hero {
    background: linear-gradient(135deg, #0F2766 0%, #1D4ED8 52%, #2563EB 100%);
    border-radius: 22px;
    padding: 36px 38px;
    color: #FFFFFF !important;
    margin-bottom: 24px;
    box-shadow: 0 16px 36px rgba(37, 99, 235, 0.22);
    display: flex;
    flex-direction: column;
    gap: 16px;
}
.aiw-hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    align-self: flex-start;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.24);
    padding: 5px 13px;
    border-radius: 999px;
    color: #FFFFFF !important;
}
.aiw-hero h2 {
    margin: 0 !important;
    font-size: 31px !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    color: #FFFFFF !important;
    line-height: 1.25 !important;
    max-width: 820px;
}
.aiw-hero p {
    margin: 0 !important;
    max-width: 780px;
    font-size: 15.5px !important;
    line-height: 1.65 !important;
    color: rgba(255, 255, 255, 0.92) !important;
}
.aiw-hero strong {
    color: #FFFFFF !important;
}
.aiw-hero-actions {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 4px;
}
.aiw-hero-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 11px 20px;
    border-radius: 12px;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    transition: transform 0.14s ease, box-shadow 0.14s ease, background 0.14s ease;
}
.aiw-hero-btn-primary {
    background: #FFFFFF;
    color: #1D4ED8 !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.18);
}
.aiw-hero-btn-primary:hover {
    transform: translateY(-1px);
    background: #F8FAFC;
}
.aiw-hero-btn-ghost {
    background: rgba(255, 255, 255, 0.14);
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.30);
}
.aiw-hero-btn-ghost:hover {
    transform: translateY(-1px);
    background: rgba(255, 255, 255, 0.22);
}

/* Home Module Grid */
.aiw-section-label {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary) !important;
    margin: 4px 0 12px 2px;
}
.aiw-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    margin-bottom: 20px;
}
.aiw-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 22px;
    box-shadow: var(--shadow-sm);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 14px;
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
}
.aiw-clickable {
    cursor: pointer;
    user-select: none;
}
.aiw-clickable:hover,
.aiw-clickable:focus-visible {
    transform: translateY(-3px);
    border-color: var(--primary);
    box-shadow: var(--shadow-md);
    outline: none;
}
.aiw-card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 12px;
}
.aiw-icon-box {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: var(--primary-soft);
    border: 1px solid var(--primary-border);
    color: var(--primary) !important;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.aiw-card h3 {
    margin: 0 0 6px 0 !important;
    font-size: 16.5px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}
.aiw-card p {
    margin: 0 !important;
    font-size: 13.5px !important;
    line-height: 1.6 !important;
    color: var(--text-secondary) !important;
}
.aiw-card-meta {
    font-size: 12px;
    color: var(--text-secondary) !important;
    margin-top: 8px;
}
.aiw-card-cta {
    font-size: 13.5px;
    font-weight: 700;
    color: var(--primary) !important;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding-top: 10px;
    border-top: 1px solid #F1F5F9;
}
.aiw-clickable:hover .aiw-card-cta {
    color: var(--primary-hover) !important;
}

/* Elevated Studio Panels (Gradio Columns/Groups) */
.aiw-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 20px 22px !important;
    box-shadow: var(--shadow-sm) !important;
}
.aiw-config-card {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 18px 22px !important;
    margin-bottom: 18px !important;
    box-shadow: var(--shadow-xs) !important;
}
.aiw-panel-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin: 0 0 2px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.aiw-panel-caption {
    font-size: 12.5px;
    color: var(--text-secondary) !important;
    margin: 0 0 12px 0;
}

/* TTS Audio Summary & Status Card */
.aiw-audio-status {
    border: 1px solid var(--border);
    background: var(--bg);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.aiw-audio-status-ready {
    border-color: var(--success-border);
    background: #F6FEF9;
}
.aiw-audio-status-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
}
.aiw-audio-filename {
    font-size: 13.5px;
    font-weight: 700;
    color: var(--text-primary) !important;
    font-family: "Cascadia Code", "Consolas", monospace;
}
.aiw-audio-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.aiw-char-counter {
    font-size: 12.5px;
    color: var(--text-secondary) !important;
    text-align: right;
    padding: 2px 2px 6px 2px;
}

/* Chat Workspace Width */
.aiw-chat-shell {
    max-width: 1040px !important;
    margin: 0 auto !important;
}

/* System Monitoring Cards */
.aiw-sys-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 16px;
}
.aiw-sys-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 16px 18px;
    box-shadow: var(--shadow-xs);
}
.aiw-sys-card h4 {
    margin: 0 0 12px 0 !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.aiw-sys-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 10px;
    padding: 7px 0;
    border-top: 1px solid #F1F5F9;
    font-size: 12.5px;
}
.aiw-sys-row:first-of-type {
    border-top: none;
    padding-top: 0;
}
.aiw-sys-key {
    font-weight: 600;
    color: var(--text-primary) !important;
    flex-shrink: 0;
}
.aiw-sys-val {
    color: var(--text-secondary) !important;
    text-align: right;
    word-break: break-word;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 11px !important;
    padding: 10px 20px !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease !important;
}
button.primary:hover:not(:disabled) {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.34) !important;
    filter: brightness(1.03) !important;
}
button.secondary {
    background: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    border-radius: 11px !important;
    transition: border-color 0.14s ease, background 0.14s ease, color 0.14s ease !important;
}
button.secondary:hover:not(:disabled) {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
    background: var(--primary-soft) !important;
}

/* Accordions */
.accordion {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    margin-top: 14px !important;
    box-shadow: var(--shadow-xs) !important;
}

/* Sliders */
input[type="range"] {
    accent-color: var(--primary) !important;
}

/* Custom Footer */
.aiw-footer {
    margin-top: 28px;
    padding: 15px 22px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    font-size: 12.5px;
    color: var(--text-secondary) !important;
}

/* JARVIS Phase 8 Agent Components */
.aiw-jarvis-status {
    background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
    border: 1px solid #BFDBFE;
    border-radius: var(--radius-md);
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.aiw-jarvis-status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
}
.aiw-jarvis-url {
    font-size: 12px;
    font-family: "Cascadia Code", "Consolas", monospace;
    color: var(--text-secondary) !important;
    word-break: break-all;
}
.aiw-timeline {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 6px;
}
.aiw-tl-item {
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    background: var(--bg);
    border-radius: 10px;
    padding: 9px 12px;
    font-size: 12.5px;
}
.aiw-tl-item.is-warn {
    border-left-color: #D97706;
    background: #FFFBEB;
}
.aiw-tl-item.is-err {
    border-left-color: #DC2626;
    background: #FEF2F2;
}
.aiw-tl-item.is-ok {
    border-left-color: var(--success);
}
.aiw-tl-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin-bottom: 3px;
}
.aiw-tl-body {
    color: var(--text-secondary) !important;
    line-height: 1.45;
}
.aiw-sources {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 6px;
}
.aiw-source-item {
    display: block;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    background: var(--surface);
    text-decoration: none !important;
    transition: border-color 0.14s ease, background 0.14s ease;
}
.aiw-source-item:hover {
    border-color: var(--primary);
    background: var(--primary-soft);
}
.aiw-source-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--primary) !important;
    margin-bottom: 2px;
}
.aiw-source-url {
    font-size: 11.5px;
    color: var(--text-muted) !important;
    word-break: break-all;
}
.aiw-approval-box {
    border: 1.5px solid #F59E0B;
    background: #FFFBEB;
    border-radius: var(--radius-md);
    padding: 14px 16px;
    margin-top: 10px;
}
.aiw-approval-box.is-sensitive {
    border-color: #EF4444;
    background: #FEF2F2;
}
.aiw-approval-title {
    font-size: 14px;
    font-weight: 700;
    color: #991B1B !important;
    margin-bottom: 6px;
}
.aiw-approval-detail {
    font-size: 12.5px;
    color: var(--text-primary) !important;
    line-height: 1.5;
}
.aiw-footer strong {
    color: var(--text-primary) !important;
}
.aiw-footer-stack {
    color: var(--primary) !important;
    font-weight: 600;
}

/* Responsive Breakpoints: 1024px, 768px, 390px */
@media (max-width: 1080px) {
    .gradio-container {
        padding: 14px 18px 28px 18px !important;
    }
    .aiw-grid,
    .aiw-sys-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 12px 14px 24px 14px !important;
    }
    .aiw-header {
        padding: 12px 15px;
    }
    .aiw-hero {
        padding: 24px 22px;
    }
    .aiw-hero h2 {
        font-size: 23px !important;
    }
    .aiw-grid,
    .aiw-sys-grid {
        grid-template-columns: 1fr;
    }
    .aiw-page-header {
        padding: 16px 18px;
    }
    .aiw-panel,
    .aiw-config-card {
        padding: 15px 16px !important;
    }
}

@media (max-width: 480px) {
    .gradio-container {
        padding: 10px 10px 20px 10px !important;
    }
    .tabs > .tab-nav > button {
        padding: 8px 12px !important;
        font-size: 13px !important;
    }
    .aiw-brand-name {
        font-size: 16px;
    }
    .aiw-brand-tagline {
        display: none;
    }
    .aiw-hero-actions {
        flex-direction: column;
        align-items: stretch;
    }
    .aiw-hero-btn {
        justify-content: center;
    }
}
"""

_ICONS: dict[str, str] = {
    "nlp": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M4 7V4h16v3"/><path d="M9 20h6"/><path d="M12 4v16"/></svg>'
    ),
    "tts": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>'
        '<path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>'
    ),
    "asr": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>'
        '<path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>'
    ),
    "diar": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
        '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'
    ),
    "llm": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>'
    ),
    "sys": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/>'
        '<line x1="12" y1="17" x2="12" y2="21"/></svg>'
    ),
    "jarvis": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3"/>'
        '<line x1="12" y1="3" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="21"/>'
        '<line x1="3" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="21" y2="12"/></svg>'
    ),
}


def collect_ui_readiness(settings: Settings) -> dict[str, Any]:
    """Compute lightweight filesystem/config readiness without loading any ML models."""
    from thai_offline_ai.adapters.asr import model_is_available as typhoon_ready
    from thai_offline_ai.adapters.diarization import model_is_available as diar_ready
    from thai_offline_ai.adapters.kokoro_tts import kokoro_model_is_available
    from thai_offline_ai.adapters.llm import find_ollama_executable
    from thai_offline_ai.adapters.openthai_asr import resolve_available_precision
    from thai_offline_ai.adapters.tts import model_is_available as lunarlist_ready
    from thai_offline_ai.agent.browser.browser_manager import playwright_is_installed

    openthai_prec = resolve_available_precision(
        settings.asr.openthai_model_dir, settings.asr.openthai_precision
    )
    return {
        "nlp_enabled": settings.nlp.enabled,
        "tts_enabled": settings.tts.enabled,
        "kokoro_ready": kokoro_model_is_available(settings.tts.kokoro_model_dir),
        "lunarlist_ready": lunarlist_ready(settings.tts.model_dir),
        "asr_enabled": settings.asr.enabled,
        "openthai_precision": openthai_prec,
        "typhoon_ready": typhoon_ready(settings.asr.model_dir),
        "diar_enabled": settings.diarization.enabled,
        "diar_ready": diar_ready(settings.diarization.model_dir, settings.diarization.precision),
        "llm_enabled": settings.llm.enabled,
        "ollama_found": bool(find_ollama_executable()),
        "playwright_ready": playwright_is_installed(),
    }


def build_theme(gr: Any) -> Any:
    """Create the clean cobalt-blue SaaS theme with local system fonts only.

    Pins both light and ``*_dark`` tokens to the light palette so Windows/browser
    Dark Mode never renders white text on white surfaces.
    """
    cobalt = gr.themes.Color(
        name="workbench_cobalt",
        c50="#EFF6FF",
        c100="#DBEAFE",
        c200="#BFDBFE",
        c300="#93C5FD",
        c400="#60A5FA",
        c500="#2563EB",
        c600="#1D4ED8",
        c700="#1E40AF",
        c800="#1E3A8A",
        c900="#172554",
        c950="#0F172A",
    )
    return gr.themes.Soft(
        primary_hue=cobalt,
        secondary_hue=cobalt,
        neutral_hue="slate",
        radius_size="lg",
        font=SYSTEM_SANS_FONTS,
        font_mono=SYSTEM_MONO_FONTS,
    ).set(
        body_background_fill="#F8FAFC",
        body_background_fill_dark="#F8FAFC",
        body_text_color="#0F172A",
        body_text_color_dark="#0F172A",
        body_text_color_subdued="#64748B",
        body_text_color_subdued_dark="#64748B",
        background_fill_primary="#FFFFFF",
        background_fill_primary_dark="#FFFFFF",
        background_fill_secondary="#F8FAFC",
        background_fill_secondary_dark="#F8FAFC",
        border_color_primary="#E2E8F0",
        border_color_primary_dark="#E2E8F0",
        block_background_fill="#FFFFFF",
        block_background_fill_dark="#FFFFFF",
        block_border_color="#E2E8F0",
        block_border_color_dark="#E2E8F0",
        block_border_width="1px",
        block_border_width_dark="1px",
        block_radius="14px",
        block_shadow="0 1px 2px rgba(15, 23, 42, 0.04)",
        block_shadow_dark="0 1px 2px rgba(15, 23, 42, 0.04)",
        block_label_background_fill="#FFFFFF",
        block_label_background_fill_dark="#FFFFFF",
        block_label_text_color="#0F172A",
        block_label_text_color_dark="#0F172A",
        block_title_text_color="#0F172A",
        block_title_text_color_dark="#0F172A",
        block_title_text_weight="600",
        block_info_text_color="#64748B",
        block_info_text_color_dark="#64748B",
        panel_background_fill="#FFFFFF",
        panel_background_fill_dark="#FFFFFF",
        panel_border_color_dark="#E2E8F0",
        input_background_fill="#FFFFFF",
        input_background_fill_dark="#FFFFFF",
        input_background_fill_hover="#FFFFFF",
        input_background_fill_hover_dark="#FFFFFF",
        input_background_fill_focus="#FFFFFF",
        input_background_fill_focus_dark="#FFFFFF",
        input_border_color="#CBD5E1",
        input_border_color_dark="#CBD5E1",
        input_border_color_focus="#2563EB",
        input_border_color_focus_dark="#2563EB",
        input_placeholder_color="#94A3B8",
        input_placeholder_color_dark="#94A3B8",
        button_primary_background_fill="#2563EB",
        button_primary_background_fill_dark="#2563EB",
        button_primary_background_fill_hover="#1D4ED8",
        button_primary_background_fill_hover_dark="#1D4ED8",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
        button_secondary_background_fill="#FFFFFF",
        button_secondary_background_fill_dark="#FFFFFF",
        button_secondary_background_fill_hover="#EFF6FF",
        button_secondary_background_fill_hover_dark="#EFF6FF",
        button_secondary_text_color="#0F172A",
        button_secondary_text_color_dark="#0F172A",
        button_secondary_text_color_hover="#2563EB",
        button_secondary_text_color_hover_dark="#2563EB",
        checkbox_label_background_fill="#F8FAFC",
        checkbox_label_background_fill_dark="#F8FAFC",
        checkbox_label_background_fill_hover="#EFF6FF",
        checkbox_label_background_fill_hover_dark="#EFF6FF",
        checkbox_label_background_fill_selected="#EFF6FF",
        checkbox_label_background_fill_selected_dark="#EFF6FF",
        checkbox_label_text_color="#0F172A",
        checkbox_label_text_color_dark="#0F172A",
        checkbox_label_text_color_selected="#1D4ED8",
        checkbox_label_text_color_selected_dark="#1D4ED8",
        accordion_text_color="#0F172A",
        accordion_text_color_dark="#0F172A",
        table_text_color="#0F172A",
        table_text_color_dark="#0F172A",
        table_even_background_fill="#FFFFFF",
        table_even_background_fill_dark="#FFFFFF",
        table_odd_background_fill="#F8FAFC",
        table_odd_background_fill_dark="#F8FAFC",
        code_background_fill="#EFF6FF",
        code_background_fill_dark="#EFF6FF",
        link_text_color="#2563EB",
        link_text_color_dark="#2563EB",
    )


def render_navbar(settings: Settings, readiness: dict[str, Any] | None = None) -> str:
    info = readiness if readiness is not None else collect_ui_readiness(settings)
    offline_badge = (
        '<span class="aiw-pill aiw-pill-success"><span class="aiw-dot"></span>Offline 100%</span>'
        if settings.app.offline_runtime
        else '<span class="aiw-pill aiw-pill-warn">Offline Guard Off</span>'
    )
    host_port = escape(f"{settings.app.host}:{settings.app.port}")
    tts_label = "Kokoro 24kHz" if info["kokoro_ready"] else ("Lunarlist" if info["lunarlist_ready"] else "TTS Setup")
    asr_label = (
        f"OpenThai 2.0 ({settings.asr.openthai_prompt})"
        if info["openthai_precision"]
        else ("Typhoon ASR" if info["typhoon_ready"] else "ASR Setup")
    )
    return f"""
    <div class="aiw-header">
      <div class="aiw-brand" role="button" tabindex="0" data-target-tab="Home" title="กลับหน้าแรก (Home)">
        <div class="aiw-logo" aria-hidden="true">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <rect x="3.5" y="9" width="3" height="6" rx="1.5" fill="#93C5FD"/>
            <rect x="8.5" y="5" width="3" height="14" rx="1.5" fill="#FFFFFF"/>
            <rect x="13.5" y="7.5" width="3" height="9" rx="1.5" fill="#BFDBFE"/>
            <rect x="18.5" y="10" width="2.5" height="4" rx="1.25" fill="#FFFFFF"/>
          </svg>
        </div>
        <div>
          <div class="aiw-brand-name">Thai Offline <span>AI Workbench</span></div>
          <div class="aiw-brand-tagline">Private Local AI Suite · Thai &amp; English · CPU-First</div>
        </div>
      </div>
      <div class="aiw-header-status">
        {offline_badge}
        <span class="aiw-pill aiw-pill-blue">Local: {escape(tts_label)} · {escape(asr_label)}</span>
        <span class="aiw-pill">{host_port}</span>
      </div>
    </div>
    """


def render_page_header(
    title: str,
    subtitle: str,
    badges: Sequence[tuple[str, str]] = (),
) -> str:
    """Render a consistent studio page header. ``badges`` are ``(text, tone)`` pairs."""
    tone_class = {
        "success": "aiw-pill aiw-pill-success",
        "blue": "aiw-pill aiw-pill-blue",
        "warn": "aiw-pill aiw-pill-warn",
        "err": "aiw-pill aiw-pill-err",
        "neutral": "aiw-pill",
    }
    badge_html = "".join(
        f'<span class="{tone_class.get(tone, "aiw-pill")}">{escape(label)}</span>'
        for label, tone in badges
    )
    return (
        '<div class="aiw-page-header">'
        f'<div><h2 class="aiw-page-title">{escape(title)}</h2>'
        f'<p class="aiw-page-sub">{escape(subtitle)}</p></div>'
        f'<div class="aiw-page-badges">{badge_html}</div>'
        "</div>"
    )


def render_home_html(settings: Settings, readiness: dict[str, Any] | None = None) -> str:
    info = readiness if readiness is not None else collect_ui_readiness(settings)

    def pill(text: str, tone: str) -> str:
        cls = {
            "success": "aiw-pill aiw-pill-success",
            "warn": "aiw-pill aiw-pill-warn",
            "neutral": "aiw-pill",
        }.get(tone, "aiw-pill")
        return f'<span class="{cls}">{escape(text)}</span>'

    nlp_badge = pill("พร้อมใช้งาน", "success") if info["nlp_enabled"] else pill("ปิดอยู่", "neutral")

    if not info["tts_enabled"]:
        tts_badge = pill("ปิดอยู่", "neutral")
    elif info["kokoro_ready"] or info["lunarlist_ready"]:
        tts_badge = pill("พร้อมใช้งาน", "success")
    else:
        tts_badge = pill("รอติดตั้งโมเดล", "warn")

    if not info["asr_enabled"]:
        asr_badge = pill("ปิดอยู่", "neutral")
    elif info["openthai_precision"] or info["typhoon_ready"]:
        asr_badge = pill("พร้อมใช้งาน", "success")
    else:
        asr_badge = pill("รอติดตั้งโมเดล", "warn")

    if not info["diar_enabled"]:
        diar_badge = pill("ปิดอยู่", "neutral")
    elif info["diar_ready"]:
        diar_badge = pill("พร้อมใช้งาน", "success")
    else:
        diar_badge = pill("รอติดตั้งโมเดล", "warn")

    if not info["llm_enabled"]:
        llm_badge = pill("ปิดอยู่", "neutral")
    elif info["ollama_found"]:
        llm_badge = pill("เชื่อมต่อ Local", "success")
    else:
        llm_badge = pill("ไม่พบ Ollama", "warn")

    tts_engines = ", ".join(
        name
        for name, ok in (
            ("Kokoro 24 kHz", info["kokoro_ready"]),
            ("Lunarlist ONNX", info["lunarlist_ready"]),
        )
        if ok
    ) or "ยังไม่ได้ดาวน์โหลดแคชโมเดล"

    asr_engines = ", ".join(
        name
        for name, ok in (
            (f"OpenThai 2.0 ({info['openthai_precision']})", bool(info["openthai_precision"])),
            ("Typhoon ASR", info["typhoon_ready"]),
        )
        if ok
    ) or "ยังไม่ได้ดาวน์โหลดแคชโมเดล"

    return f"""
    <div class="aiw-hero">
      <div class="aiw-hero-kicker"><span class="aiw-dot"></span>On-Device Thai &amp; English AI · Zero Cloud</div>
      <h2>สตูดิโอ AI ภาษาไทยและอังกฤษบนเครื่องคุณ<br/>ครบทั้งข้อความ เสียงพูด ถอดเสียง และแชตส่วนตัว</h2>
      <p>
        ประมวลผลทั้งหมดบนเครื่องแบบ <strong>CPU-First</strong> พร้อมระบบป้องกันการเชื่อมต่อออกนอกเครือข่าย
        โหลดโมเดลแบบ <strong>Lazy Loading</strong> เฉพาะตอนกดใช้งาน รองรับทั้งภาษาไทยล้วนและไทยผสมอังกฤษ
      </p>
      <div class="aiw-hero-actions">
        <span class="aiw-hero-btn aiw-hero-btn-primary" role="button" tabindex="0" data-target-tab="JARVIS AI Assistant">
          เปิด JARVIS AI Assistant →
        </span>
        <span class="aiw-hero-btn aiw-hero-btn-ghost" role="button" tabindex="0" data-target-tab="TTS">
          เปิด TTS Studio →
        </span>
        <span class="aiw-hero-btn aiw-hero-btn-ghost" role="button" tabindex="0" data-target-tab="ASR">
          เปิด ASR Workspace →
        </span>
        <span class="aiw-hero-btn aiw-hero-btn-ghost" role="button" tabindex="0" data-target-tab="Local LLM">
          คุยกับ Local LLM →
        </span>
      </div>
    </div>

    <div class="aiw-section-label">AI Modules · คลิกการ์ดเพื่อเปิดเครื่องมือ</div>
    <div class="aiw-grid">
      <div class="aiw-card aiw-clickable" role="button" tabindex="0" data-target-tab="JARVIS AI Assistant">
        <div>
          <div class="aiw-card-head">
            <div class="aiw-icon-box">{_ICONS["jarvis"]}</div>
            {pill("Playwright พร้อม" if info.get("playwright_ready") else "Offline Agent พร้อม", "success" if info.get("playwright_ready") else "warn")}
          </div>
          <h3>JARVIS AI Assistant (Phase 8)</h3>
          <p>ผู้ช่วยส่วนตัวสั่งงานด้วยเสียง/ข้อความ วางแผนหลายขั้นตอน เปิด Browser ค้นเว็บจริงพร้อมแหล่งอ้างอิง และขออนุมัติก่อนทำรายการสำคัญ</p>
          <div class="aiw-card-meta">โหมด: <code>Offline / Online Agent</code> · <code>SQLite Memory</code></div>
        </div>
        <div class="aiw-card-cta">เปิด JARVIS AI Assistant →</div>
      </div>

      <div class="aiw-card aiw-clickable" role="button" tabindex="0" data-target-tab="TTS">
        <div>
          <div class="aiw-card-head">
            <div class="aiw-icon-box">{_ICONS["tts"]}</div>
            {tts_badge}
          </div>
          <h3>Text-to-Speech Studio</h3>
          <p>สังเคราะห์เสียงพูดไทยและไทยผสมอังกฤษ คุณภาพ 24 kHz เลือกเสียงได้ 12 สไตล์ พร้อมปรับความเร็วการพูด</p>
          <div class="aiw-card-meta">แคชในเครื่อง: <code>{escape(tts_engines)}</code></div>
        </div>
        <div class="aiw-card-cta">เปิด TTS Studio →</div>
      </div>

      <div class="aiw-card aiw-clickable" role="button" tabindex="0" data-target-tab="ASR">
        <div>
          <div class="aiw-card-head">
            <div class="aiw-icon-box">{_ICONS["asr"]}</div>
            {asr_badge}
          </div>
          <h3>Speech Recognition (ASR)</h3>
          <p>ถอดเสียงไฟล์ WAV/FLAC/MP3/OGG, ถอดเสียงแบบชุด (Batch) และไมโครโฟนสด รองรับคำไทยผสมอังกฤษ</p>
          <div class="aiw-card-meta">แคชในเครื่อง: <code>{escape(asr_engines)}</code></div>
        </div>
        <div class="aiw-card-cta">เปิด ASR Workspace →</div>
      </div>

      <div class="aiw-card aiw-clickable" role="button" tabindex="0" data-target-tab="ASR" data-open-accordion="Speaker Diarization">
        <div>
          <div class="aiw-card-head">
            <div class="aiw-icon-box">{_ICONS["diar"]}</div>
            {diar_badge}
          </div>
          <h3>Speaker Diarization</h3>
          <p>แยกช่วงเวลาผู้พูดอัตโนมัติ (<code>SPEAKER_00</code>, <code>SPEAKER_01</code>) ทั้งจากไฟล์และสตรีมสดด้วย Nemotron-3 ONNX</p>
          <div class="aiw-card-meta">โมเดล: <code>{escape(settings.diarization.model)}</code></div>
        </div>
        <div class="aiw-card-cta">เปิด Speaker Diarization →</div>
      </div>

      <div class="aiw-card aiw-clickable" role="button" tabindex="0" data-target-tab="NLP">
        <div>
          <div class="aiw-card-head">
            <div class="aiw-icon-box">{_ICONS["nlp"]}</div>
            {nlp_badge}
          </div>
          <h3>Thai &amp; English NLP</h3>
          <p>ปรับรูปแบบข้อความ ตัดคำ ตัดประโยค และสรุปสัดส่วนอักษรไทย/อังกฤษ ด้วย PyThaiNLP แบบออฟไลน์</p>
          <div class="aiw-card-meta">เอนจิน: <code>{escape(settings.nlp.word_engine)}</code> · <code>{escape(settings.nlp.sentence_engine)}</code></div>
        </div>
        <div class="aiw-card-cta">เปิด NLP Workspace →</div>
      </div>

      <div class="aiw-card aiw-clickable" role="button" tabindex="0" data-target-tab="Local LLM">
        <div>
          <div class="aiw-card-head">
            <div class="aiw-icon-box">{_ICONS["llm"]}</div>
            {llm_badge}
          </div>
          <h3>Local LLM Assistant</h3>
          <p>แชตกับโมเดลภาษาในเครื่องผ่าน Ollama ที่ <code>127.0.0.1</code> เท่านั้น ไม่บันทึกประวัติแชตลงดิสก์</p>
          <div class="aiw-card-meta">โมเดล: <code>{escape(settings.llm.model)}</code></div>
        </div>
        <div class="aiw-card-cta">เปิด Local LLM →</div>
      </div>

      <div class="aiw-card aiw-clickable" role="button" tabindex="0" data-target-tab="System">
        <div>
          <div class="aiw-card-head">
            <div class="aiw-icon-box">{_ICONS["sys"]}</div>
            {pill("Offline Guard On" if settings.app.offline_runtime else "Local Mode", "success" if settings.app.offline_runtime else "warn")}
          </div>
          <h3>System Monitor</h3>
          <p>ตรวจสอบสภาพแวดล้อม Python, CPU/GPU, สถานะแคชโมเดล และนโยบายความปลอดภัยออฟไลน์แบบไม่โหลดโมเดล</p>
          <div class="aiw-card-meta">ปลายทาง: <code>{escape(settings.app.host)}:{settings.app.port}</code></div>
        </div>
        <div class="aiw-card-cta">เปิด System Monitor →</div>
      </div>
    </div>
    """


def render_tts_audio_summary(
    *,
    filename: str | None = None,
    duration_seconds: float | None = None,
    sample_rate: int | None = None,
    model: str | None = None,
    voice: str | None = None,
    speed: float | None = None,
    elapsed_seconds: float | None = None,
) -> str:
    """Render a clean TTS output status card without exposing long absolute paths."""
    if not filename:
        return (
            '<div class="aiw-audio-status">'
            '<div class="aiw-audio-status-top">'
            '<span class="aiw-pill">ยังไม่มีเสียงที่สร้าง</span>'
            '<span style="font-size:12.5px;color:#64748B;">พิมพ์ข้อความด้านซ้ายแล้วกด <strong>สร้างเสียง</strong></span>'
            "</div></div>"
        )
    metrics: list[str] = []
    if duration_seconds is not None:
        metrics.append(f'<span class="aiw-pill aiw-pill-blue">ความยาว {duration_seconds:.2f}s</span>')
    if sample_rate is not None:
        metrics.append(f'<span class="aiw-pill">{sample_rate / 1000:.2f} kHz</span>')
    if voice:
        metrics.append(f'<span class="aiw-pill">{escape(voice)}</span>')
    if speed is not None:
        metrics.append(f'<span class="aiw-pill">{speed:.2f}x</span>')
    if model:
        metrics.append(f'<span class="aiw-pill">{escape(model)}</span>')
    if elapsed_seconds is not None:
        metrics.append(f'<span class="aiw-pill">สร้างใน {elapsed_seconds:.2f}s</span>')
    return (
        '<div class="aiw-audio-status aiw-audio-status-ready">'
        '<div class="aiw-audio-status-top">'
        '<span class="aiw-pill aiw-pill-success"><span class="aiw-dot"></span>สร้างเสียงสำเร็จ</span>'
        f'<span class="aiw-audio-filename">{escape(filename)}</span>'
        "</div>"
        f'<div class="aiw-audio-metrics">{"".join(metrics)}</div>'
        "</div>"
    )


def render_system_cards(checks: Sequence[Any], settings: Settings) -> str:
    """Organize genuine backend diagnostic checks into structured monitoring cards."""
    def badge(status: str) -> str:
        tone = {
            "OK": "aiw-pill aiw-pill-success",
            "WARN": "aiw-pill aiw-pill-warn",
            "FAIL": "aiw-pill aiw-pill-err",
            "SKIP": "aiw-pill",
        }.get(status, "aiw-pill")
        return f'<span class="{tone}">{escape(status)}</span>'

    runtime_rows: list[str] = [
        (
            '<div class="aiw-sys-row"><span class="aiw-sys-key">CPU Logical Cores</span>'
            f'<span class="aiw-sys-val">{os.cpu_count() or "-"} · {escape(platform.machine() or platform.system())}</span></div>'
        ),
    ]
    model_rows: list[str] = []
    endpoint_rows: list[str] = []

    runtime_names = {
        "Operating system",
        "Python",
        "NVIDIA GPU",
        "ONNX providers",
        "Diarization CUDA",
        "Diarization device",
        "ASR device",
    }
    endpoint_names = {
        "Offline runtime",
        "UI bind",
        "LLM URL",
        "Ollama executable",
        "Ollama service",
        "Ollama model",
        "Output directory",
        "Log level",
    }

    for check in checks:
        row = (
            f'<div class="aiw-sys-row">'
            f'<span class="aiw-sys-key">{escape(check.name)}</span>'
            f'<span class="aiw-sys-val">{badge(check.status)} {escape(check.detail)}</span>'
            f"</div>"
        )
        if check.name in runtime_names:
            runtime_rows.append(row)
        elif check.name in endpoint_names:
            endpoint_rows.append(row)
        elif check.name.startswith("Component ") or "cache" in check.name.lower() or "streaming" in check.name.lower():
            model_rows.append(row)

    if not model_rows:
        model_rows.append(
            '<div class="aiw-sys-row"><span class="aiw-sys-val">กด "ตรวจสอบระบบ" เพื่อโหลดรายละเอียด</span></div>'
        )

    ok_count = sum(1 for c in checks if c.status == "OK")
    warn_count = sum(1 for c in checks if c.status == "WARN")
    fail_count = sum(1 for c in checks if c.status == "FAIL")
    summary_pill = (
        f'<span class="aiw-pill aiw-pill-err">{fail_count} FAIL</span>'
        if fail_count
        else f'<span class="aiw-pill aiw-pill-success">{ok_count} OK · {warn_count} WARN</span>'
    )

    return f"""
    <div class="aiw-sys-grid">
      <div class="aiw-sys-card">
        <h4>Hardware &amp; Runtime {summary_pill}</h4>
        {"".join(runtime_rows)}
      </div>
      <div class="aiw-sys-card">
        <h4>Model Readiness &amp; Caches</h4>
        {"".join(model_rows)}
      </div>
      <div class="aiw-sys-card">
        <h4>Local Endpoints &amp; Privacy</h4>
        {"".join(endpoint_rows)}
      </div>
    </div>
    """


def render_jarvis_status_card(
    *,
    mode: str = "offline",
    status: str = "idle",
    browser_active: bool = False,
    browser_url: str | None = None,
    playwright_ready: bool = False,
    detail: str = "",
) -> str:
    """Render live operating mode, execution state, and visible browser status for JARVIS."""
    mode_pill = (
        '<span class="aiw-pill aiw-pill-blue"><span class="aiw-dot" style="background:#2563EB;"></span>Online Agent Mode</span>'
        if mode == "online"
        else '<span class="aiw-pill aiw-pill-success"><span class="aiw-dot"></span>Offline Mode (Local Only)</span>'
    )
    status_map = {
        "idle": ("พร้อมรับคำสั่ง", "aiw-pill"),
        "completed": ("ทำงานเสร็จสิ้น", "aiw-pill aiw-pill-success"),
        "awaiting_approval": ("รอการอนุมัติจากผู้ใช้", "aiw-pill aiw-pill-warn"),
        "cancelled": ("หยุดการทำงานแล้ว", "aiw-pill aiw-pill-warn"),
        "error": ("พบข้อผิดพลาด", "aiw-pill aiw-pill-err"),
    }
    st_label, st_cls = status_map.get(status, (status, "aiw-pill"))
    browser_pill = (
        '<span class="aiw-pill aiw-pill-blue">Browser: เปิดอยู่</span>'
        if browser_active
        else (
            '<span class="aiw-pill">Browser: ปิดอยู่</span>'
            if playwright_ready
            else '<span class="aiw-pill aiw-pill-warn">Playwright ยังไม่ติดตั้ง</span>'
        )
    )
    url_html = (
        f'<div class="aiw-jarvis-url">Active URL: {escape(browser_url)}</div>'
        if browser_url
        else '<div class="aiw-jarvis-url">ยังไม่มีหน้าเว็บที่เปิดในเซสชันนี้</div>'
    )
    detail_html = (
        f'<div style="font-size:12.5px;color:#475569;">{escape(detail)}</div>' if detail else ""
    )
    return (
        '<div class="aiw-jarvis-status">'
        f'<div class="aiw-jarvis-status-row">{mode_pill}<span class="{st_cls}">{escape(st_label)}</span>{browser_pill}</div>'
        f"{url_html}{detail_html}"
        "</div>"
    )


def render_jarvis_timeline(steps: Sequence[Any] = ()) -> str:
    """Render structured agent planning and tool execution timeline."""
    if not steps:
        return (
            '<div class="aiw-timeline">'
            '<div class="aiw-tl-item"><div class="aiw-tl-body">'
            "ยังไม่มีประวัติการทำงาน — พิมพ์คำสั่งหรือกด Push-to-Talk เพื่อเริ่มให้ JARVIS วางแผนและทำงาน"
            "</div></div></div>"
        )
    items: list[str] = []
    for step in steps:
        st = getattr(step, "status", "completed")
        tone = "is-ok" if st == "completed" else ("is-warn" if st in {"awaiting_approval", "cancelled"} else "is-err")
        tool_tag = (
            f'<span class="aiw-pill aiw-pill-blue">{escape(str(step.tool_name))}</span>'
            if getattr(step, "tool_name", None)
            else ""
        )
        items.append(
            f'<div class="aiw-tl-item {tone}">'
            f'<div class="aiw-tl-head"><span>#{step.step_number} · {escape(str(step.phase))}</span>{tool_tag}</div>'
            f'<div class="aiw-tl-body">{escape(str(step.summary))}</div>'
            "</div>"
        )
    return f'<div class="aiw-timeline">{"".join(items)}</div>'


def render_jarvis_sources(sources: Sequence[Any] = ()) -> str:
    """Render clickable external source links gathered by the browser agent."""
    if not sources:
        return (
            '<div style="font-size:12.5px;color:#64748B;padding:6px 2px;">'
            "ยังไม่มีแหล่งอ้างอิงจากเว็บไซต์ (เปิด Online Agent Mode เพื่อค้นหาเว็บจริงพร้อมลิงก์)"
            "</div>"
        )
    cards: list[str] = []
    for src in sources:
        url = str(getattr(src, "url", "") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        title = str(getattr(src, "title", "") or url)
        snippet = str(getattr(src, "snippet", "") or "")
        snip_html = (
            f'<div style="font-size:12px;color:#475569;margin-top:3px;">{escape(snippet[:180])}</div>'
            if snippet
            else ""
        )
        cards.append(
            f'<a class="aiw-source-item" href="{escape(url)}" target="_blank" rel="noopener noreferrer">'
            f'<div class="aiw-source-title">{escape(title)} ↗</div>'
            f'<div class="aiw-source-url">{escape(url)}</div>'
            f"{snip_html}</a>"
        )
    return f'<div class="aiw-sources">{"".join(cards)}</div>'


def render_jarvis_approval_card(approval: Any | None = None) -> str:
    """Render explicit action-specific approval card when Level 2/3 confirmation is required."""
    if approval is None:
        return (
            '<div style="font-size:12.5px;color:#64748B;padding:6px 2px;">'
            "ไม่มีคำขออนุมัติที่ค้างอยู่ — การกระทำระดับ Interactive / Sensitive จะหยุดรอคุณยืนยันที่นี่เสมอ"
            "</div>"
        )
    level = int(getattr(approval, "permission_level", 2))
    sens_cls = "is-sensitive" if level >= 3 else ""
    target_html = (
        f"<br/><strong>Target URL:</strong> <code>{escape(str(approval.target_url))}</code>"
        if getattr(approval, "target_url", None)
        else ""
    )
    return (
        f'<div class="aiw-approval-box {sens_cls}">'
        f'<div class="aiw-approval-title">ต้องขออนุญาตก่อนดำเนินการ (Level {level})</div>'
        '<div class="aiw-approval-detail">'
        f"<strong>เครื่องมือ:</strong> <code>{escape(str(approval.tool_name))}</code><br/>"
        f"<strong>การกระทำที่เสนอ:</strong> {escape(str(approval.action_summary))}<br/>"
        f"<strong>เหตุผลด้านความปลอดภัย:</strong> {escape(str(approval.reason))}"
        f"{target_html}"
        "</div></div>"
    )


def render_footer(settings: Settings) -> str:
    host_port = escape(f"http://{settings.app.host}:{settings.app.port}")
    return f"""
    <div class="aiw-footer">
      <div><strong>Thai Offline AI Workbench</strong> · Local-First &amp; Private ({host_port})</div>
      <div class="aiw-footer-stack">PyThaiNLP · Kokoro Thai 24 kHz · OpenThai 2.0 · Typhoon ASR · Nemotron Diarization · Ollama · JARVIS Agent</div>
    </div>
    """
