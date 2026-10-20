#!/usr/bin/env python3
"""Shared conformal-frame labels for the genus-two partition functions."""

UNIT_AREA_BERGMAN_FRAME = "bergman:unit-area"
THETA_PLUMBING_FRAME = "plumbing:theta"
GLASSES_PLUMBING_FRAME = "plumbing:glasses"

MATTER_CONFORMAL_FRAMES = frozenset(
    {
        UNIT_AREA_BERGMAN_FRAME,
        THETA_PLUMBING_FRAME,
        GLASSES_PLUMBING_FRAME,
    }
)
