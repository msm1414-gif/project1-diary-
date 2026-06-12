"""AI-powered model router: pick Sonnet, Opus, or Fable for a given task."""

from model_router.router import (
    CANDIDATES,
    ModelRecommendation,
    recommend_model,
)

__all__ = ["CANDIDATES", "ModelRecommendation", "recommend_model"]
__version__ = "0.1.0"
