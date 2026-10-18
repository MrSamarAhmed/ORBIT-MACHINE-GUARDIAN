"""Configuration for the AMD Token Factory advisory client.

Credentials come from the process environment or a git-ignored ``.env``
file — never from source. The API key is never logged, printed, or
written into any artifact: ``AmdConfig.redacted()`` exists so callers
can record which model/endpoint was used without exposing the secret.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

ENV_API_KEY = "RADEON_API_KEY"
ENV_MODEL = "RADEON_MODEL"
ENV_BASE_URL = "RADEON_BASE_URL"

DEFAULT_MODEL = "DeepSeek-V4-Flash"
DEFAULT_BASE_URL = "https://developer.amd.com.cn/radeon/api/v1"
# Live-verified 2026-10-18: the Token Factory endpoint answered a real
# advisory request in >30s but within 120s, so the default timeout must
# accommodate slow reasoning-model responses.
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_DOTENV_PATH = Path(".env")


def _read_dotenv(path: Path) -> Dict[str, str]:
    """Parse ``KEY=VALUE`` lines from a local ``.env`` file, if present.

    Values are returned but never logged. A missing or unreadable file
    yields an empty mapping so configuration degrades to environment
    variables only.
    """
    values: Dict[str, str] = {}
    try:
        if not path.is_file():
            return values
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        return {}
    return values


@dataclass(frozen=True)
class AmdConfig:
    """Immutable AMD endpoint configuration."""

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("api_key must be a non-empty string")
        if not self.model:
            raise ValueError("model must be a non-empty string")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must be an http(s) URL, got {self.base_url!r}")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        dotenv_path: Optional[Path] = None,
    ) -> Optional["AmdConfig"]:
        """Build a config from the environment, falling back to ``.env``.

        Real environment variables take precedence over ``.env`` so an
        explicit export always wins. Returns ``None`` when no API key is
        configured, letting callers run the deterministic pipeline
        unchanged instead of failing.
        """
        if dotenv_path is None:
            # Resolved at call time so tests can redirect the default path.
            dotenv_path = DEFAULT_DOTENV_PATH
        environ = dict(os.environ if env is None else env)
        file_values = _read_dotenv(dotenv_path)

        def lookup(name: str) -> str:
            return (environ.get(name) or file_values.get(name) or "").strip()

        api_key = lookup(ENV_API_KEY)
        if not api_key:
            return None
        try:
            timeout_s = float(lookup("RADEON_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
        except ValueError:
            timeout_s = DEFAULT_TIMEOUT_S
        return cls(
            api_key=api_key,
            model=lookup(ENV_MODEL) or DEFAULT_MODEL,
            base_url=lookup(ENV_BASE_URL) or DEFAULT_BASE_URL,
            timeout_s=timeout_s,
        )

    def redacted(self) -> Dict[str, object]:
        """Endpoint identity without the secret, safe to record."""
        return {
            "provider": "amd_token_factory",
            "model": self.model,
            "base_url": self.base_url,
            "timeout_s": self.timeout_s,
            "api_key": "[REDACTED]",
        }
