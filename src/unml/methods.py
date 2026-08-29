"""Supported unlearning-method identifiers shared by entry points and training."""
from __future__ import annotations


UNLEARNING_METHODS = (
    "ga_kl",
    "counterfactual_rebind",
    "entropy_rebind",
    "h_tgsd",
    "h_tgsd_no_sibling_preservation",
)

H_TGSD_METHODS = frozenset(
    {"h_tgsd", "h_tgsd_no_sibling_preservation"}
)


METHOD_RESEARCH_ROLES = {
    "ga_kl": "comparison",
    "counterfactual_rebind": "contribution",
    "entropy_rebind": "exploratory",
    "h_tgsd": "contribution",
    "h_tgsd_no_sibling_preservation": "ablation",
}


def research_role(method: str) -> str:
    try:
        return METHOD_RESEARCH_ROLES[method]
    except KeyError as exc:
        raise ValueError(f"Unknown unlearning method: {method!r}") from exc
