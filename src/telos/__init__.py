"""
TEL-OS SDK - Main Entry Point

The TEL-OS SDK provides governance and control mechanisms for LLMs,
implementing the TELOS V2 Governor with standardized transformer architecture.
"""

from .core.engine import TELOS_V2_Governor, create_engine

__all__ = ['TELOS_V2_Governor', 'create_engine']
__version__ = '2.1.1'