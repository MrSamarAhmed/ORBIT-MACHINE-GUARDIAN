"""AMD Token Factory advisory integration (optional, never authoritative).

Importing this package does not import the ``openai`` SDK or make any
network call: the client is built lazily inside ``AmdAdvisor.assess``.
"""
from orbit.amd.advisor import (
    ALLOWED_CONFIDENCE,
    PROVIDER,
    REQUIRED_FIELDS,
    AmdAdvisor,
    build_evidence_payload,
    validate_assessment,
)
from orbit.amd.config import DEFAULT_BASE_URL, DEFAULT_MODEL, AmdConfig

__all__ = [
    "ALLOWED_CONFIDENCE",
    "AmdAdvisor",
    "AmdConfig",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "PROVIDER",
    "REQUIRED_FIELDS",
    "build_evidence_payload",
    "validate_assessment",
]
