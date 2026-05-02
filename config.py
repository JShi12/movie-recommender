"""Compatibility shim for older imports.

New code should import shared configuration from ``shared.config`` and
retrieval-specific configuration from ``retrieval.config``.
"""

from retrieval.config import *  # noqa: F403
