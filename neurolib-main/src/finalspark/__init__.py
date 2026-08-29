"""Read-only integration helpers for FinalSpark NeuroPlatform data."""

from .client import (
    FinalSparkClient,
    FinalSparkIntegrationError,
    prepare_pse_from_finalspark,
)

__all__ = [
    "FinalSparkClient",
    "FinalSparkIntegrationError",
    "prepare_pse_from_finalspark",
]
