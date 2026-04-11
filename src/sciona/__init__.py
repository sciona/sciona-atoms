"""Temporary compatibility shim for the namespace-pilot repo.

This keeps the new sibling repo importable while the main matcher repo still owns a
classic `sciona` package. Once matcher is converted to a namespace-compatible root,
this file should be removed so `sciona` becomes a true PEP 420 namespace package.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
