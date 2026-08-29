"""Compatibility import path for the campaign geometry validation API."""

from agents.geometry import (
    DEFAULT_MIN_DISTANCE_ANGSTROM,
    GeometryFailureCode,
    GeometryValidationCode,
    GeometryValidationResult,
    GeometryValidator,
    validate_geometry,
)

__all__ = [
    "DEFAULT_MIN_DISTANCE_ANGSTROM",
    "GeometryFailureCode",
    "GeometryValidationCode",
    "GeometryValidationResult",
    "GeometryValidator",
    "validate_geometry",
]
