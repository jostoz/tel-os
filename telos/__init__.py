"""
TEL-OS: Governance Engine for LLM Safety

A lightweight, effective defense system against jailbreak attacks.
Validated with StrongREJECT benchmark.

Example:
    >>> from telos import TELGovernor
    >>> governor = TELGovernor()
    >>> governor.attach(model)
"""

from .governor import TELGovernor, TELConfig, create_governor

__version__ = "2.0.0"
__author__ = "jostoz"

__all__ = [
    "TELGovernor",
    "TELConfig", 
    "create_governor",
]