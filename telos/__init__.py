"""
TEL-OS: Tensor-Based Ethelial Layer for Operational Safeguards

A mechanistic interpretability middleware for AI governance.
"""

__version__ = "2.1.1"

# Core exports (lazy import to avoid transformer_lens conflict)
# from telos.core.engine import TelosGovernanceEngine
# from telos.core.standardized import TelosStandardizedEngine
from telos.core.hooks import GovernanceHookRegistry, HookBuilder

# Intervention exports
from telos.interventions.love_equation import LoveEquationGovernor, create_love_governor
from telos.interventions.glp_refiner import AegisLatentRefiner, create_latent_refiner
from telos.interventions.metabolic import MetabolicAuditor
from telos.interventions.capping_config import CappingConfig
from telos.interventions.assistant_axis import AssistantAxis
from telos.interventions.steering_config import SteeringConfig, Intervention
from telos.interventions.feature_steering import FeatureSteering

# Discovery exports
from telos.discovery.neuronpedia import NeuronpediaClient, Feature, Model
from telos.discovery.soul_builder import SoulScout, load_or_build_soul_vectors
from telos.discovery.circuit_breaker import CircuitBreaker, InterventionSafety
from telos.discovery.red_team import TELOSRedTeam

# Governance exports (SOTA v2.1.1-REGEX — XP-17/18/19 validated)
from telos.governance.telos_v21_stable import TELOSV21Stable, TELOSV21Config, create_v21_stable_governor

# Model exports
from telos.model.registry import SAEVectorRegistry, SAERelease
from telos.model.loaders import load_sae_w_dec

__all__ = [
    # Core
    "TelosGovernanceEngine",
    "TelosStandardizedEngine",
    "GovernanceHookRegistry",
    "HookBuilder",
    # Interventions
    "LoveEquationGovernor",
    "create_love_governor",
    "AegisLatentRefiner",
    "create_latent_refiner",
    "MetabolicAuditor",
    "CappingConfig",
    "AssistantAxis",
    "SteeringConfig",
    "Intervention",
    "FeatureSteering",
    # Discovery
    "NeuronpediaClient",
    "Feature",
    "Model",
    "SoulScout",
    "load_or_build_soul_vectors",
    "CircuitBreaker",
    "InterventionSafety",
    "TELOSRedTeam",
    # Model
    "SAEVectorRegistry",
    "SAERelease",
    # Governance (SOTA)
    "TELOSV21Stable",
    "TELOSV21Config",
    "create_v21_stable_governor",
]
