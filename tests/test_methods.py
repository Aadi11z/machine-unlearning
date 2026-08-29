from __future__ import annotations

import pytest

from unml.methods import METHOD_RESEARCH_ROLES, research_role


def test_every_supported_method_has_one_research_role() -> None:
    assert set(METHOD_RESEARCH_ROLES) >= {
        "ga_kl",
        "counterfactual_rebind",
        "entropy_rebind",
        "h_tgsd",
        "h_tgsd_no_sibling_preservation",
    }
    assert research_role("h_tgsd_no_sibling_preservation") == "ablation"


def test_unknown_method_role_fails() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        research_role("unknown")


def test_retain_only_is_not_a_supported_unlearning_method() -> None:
    assert "retain_only" not in METHOD_RESEARCH_ROLES
    with pytest.raises(ValueError, match="Unknown"):
        research_role("retain_only")
