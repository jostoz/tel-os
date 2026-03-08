"""
TEL-OS v2.1.1-REGEX: Production-Ready Governance Engine for LLM Safety

A state-of-the-art defense system against jailbreak attacks.
Validated on 993+ prompts with 0.76% average ASR.

New in v2.1.1-REGEX:
- Refusal Suppression Filter with regex case-insensitive matching
- 23 keywords for comprehensive attack detection
- Adaptive thresholds (0.05 base, 0.03 for attacks)

Example:
    >>> from telos import TELOSV21Stable, TELOSV21Config
    >>> config = TELOSV21Config()
    >>> governor = TELOSV21Stable(config, device="cuda")
    >>> hooks = governor.register_hooks(model)
    >>> result = governor.pre_process(prompt, tokenizer)
"""

from .governor import (
    TELOSV21Stable,
    TELOSV21Config,
    create_v21_stable_governor,
    VALIDATION_RESULTS,
)

__version__ = "2.1.1-REGEX"
__author__ = "TEL-OS Team"

__all__ = [
    "TELOSV21Stable",
    "TELOSV21Config",
    "create_v21_stable_governor",
    "VALIDATION_RESULTS",
]
