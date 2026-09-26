"""Process-local runtime safeguards; importing this module does not block setup."""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
from urllib.parse import urlsplit


class PrivacyError(PermissionError):
    """An operation would leave this computer during offline runtime."""


def is_loopback_host(host: object) -> bool:
    """Validate literal loopback addresses without resolving arbitrary DNS names."""
    if not isinstance(host, str) or not host or host != host.strip():
        return False
    if host.lower() == "localhost":
        return True
    if "%" in host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


def is_local_url(url: object) -> bool:
    """Accept well-formed loopback HTTP(S) base URLs without credentials or queries."""
    if not isinstance(url, str) or not url or any(c.isspace() or ord(c) < 32 for c in url):
        return False
    if "\\" in url:
        return False
    try:
        parsed = urlsplit(url)
        port = parsed.port
        return (
            parsed.scheme in {"http", "https"}
            and is_loopback_host(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and (port is None or 1 <= port <= 65535)
            and not parsed.netloc.endswith(":")
        )
    except ValueError:
        return False


def _network_audit(event: str, args: tuple[object, ...]) -> None:
    """Reject external Python socket traffic, including redirects, proxies and DNS."""
    host: object
    if event in {"socket.connect", "socket.bind", "socket.sendto"}:
        sock, address = args[0], args[-1]
        if getattr(sock, "family", None) not in {socket.AF_INET, socket.AF_INET6}:
            return
        host = address[0] if isinstance(address, tuple) and address else None
    elif event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
        host = args[0]
    elif event == "socket.sendmsg":
        sock, address = args[0], args[-1]
        if address is None or getattr(sock, "family", None) not in {
            socket.AF_INET,
            socket.AF_INET6,
        }:
            return
        host = address[0] if isinstance(address, tuple) and address else None
    else:
        return
    if not is_loopback_host(host):
        raise PrivacyError(
            "Offline runtime blocked a non-loopback network operation. "
            "Use 127.0.0.1/localhost; install dependencies separately during setup."
        )


_guard_installed = False


def configure_privacy(offline_runtime: bool = True) -> None:
    """Disable supported telemetry/downloads before importing backend libraries.

    The offline audit hook is irreversible for this process. It covers Python
    sockets, not arbitrary native libraries or child processes; this is not an
    operating-system firewall. Phase 1 does not invoke optional inference code.
    """
    global _guard_installed
    os.environ.update(
        {
            "GRADIO_ANALYTICS_ENABLED": "False",
            # Gradio also reads these at import time, before launch arguments apply.
            "GRADIO_SERVER_NAME": "127.0.0.1",
            "GRADIO_SHARE": "False",
            "GRADIO_ROOT_PATH": "",
            "GRADIO_SSR_MODE": "False",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DO_NOT_TRACK": "1",
            "PYTHAINLP_OFFLINE": "1",
            "PYTHAINLP_READ_ONLY": "1",
            "PYTHAINLP_ALLOW_UNSAFE_PICKLE": "0",
        }
    )
    # The installed PyThaiNLP rejects the legacy and current variables together.
    os.environ.pop("PYTHAINLP_READ_MODE", None)
    # Gradio's own localhost startup checks use HTTPX's environment configuration.
    # Never send those checks through an inherited HTTP(S)/ALL_PROXY.
    for variable in ("NO_PROXY", "no_proxy"):
        os.environ[variable] = "localhost,127.0.0.1,::1"
    if offline_runtime and not _guard_installed:
        sys.addaudithook(_network_audit)
        _guard_installed = True
