from __future__ import annotations

import pytest

from scripts.run_interface import _runner_mode


def test_hosted_interface_requires_complete_modal_configuration() -> None:
    with pytest.raises(SystemExit, match="requires UNML_MODAL_URL"):
        _runner_mode(hosted=True, modal_url=None, modal_secret=None)
    with pytest.raises(SystemExit, match="configured together"):
        _runner_mode(hosted=False, modal_url="https://worker", modal_secret=None)


def test_local_interface_keeps_explicit_cpu_fallback() -> None:
    assert _runner_mode(hosted=False, modal_url=None, modal_secret=None) == "subprocess"
    assert _runner_mode(hosted=True, modal_url="https://worker", modal_secret="s") == "modal"
