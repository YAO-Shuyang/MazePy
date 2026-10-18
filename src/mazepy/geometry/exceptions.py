"""Errors raised by the geometry API."""


class GeometryError(ValueError):
    """An environment has malformed or inconsistent geometry."""


class InvalidPointError(GeometryError):
    """A path endpoint is not strictly inside navigable free space."""


class NoPathError(GeometryError):
    """No wall-respecting path connects the requested endpoints."""
