"""Tests never need external network access or optional inference packages."""

import os

import pytest

from thai_offline_ai.privacy import configure_privacy

configure_privacy()


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    for name in list(os.environ):
        if name.startswith("THAI_AI_"):
            monkeypatch.delenv(name)
