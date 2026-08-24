"""Supported unlearning-method identifiers shared by entry points and training."""
from __future__ import annotations


UNLEARNING_METHODS = (
    "retain_only",
    "ga_kl",
    "counterfactual_rebind",
    "entropy_rebind",
    "h_tgsd",
    "h_tgsd_no_sibling_preservation",
)

H_TGSD_METHODS = frozenset(
    {"h_tgsd", "h_tgsd_no_sibling_preservation"}
)
