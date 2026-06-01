"""Tests for build_config_model (M2) using a fake LLM invoke (no network)."""

import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.fortigate_config_builder import build_config_model, normalize_fortios_version  # noqa: E402
from app.fortigate_render_models import FortiGateConfigModel  # noqa: E402


class FakeInvoke:
    """Returns queued responses in order; records how many times it was called."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def __call__(self, system_text, human_text, run_name):
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return ""


VALID = '{"fortigate_config": {"hostname": "x", "interfaces": [{"name": "port1", "role": "wan", "ip": "1.1.1.1 255.255.255.0", "allowaccess": ["ping"]}]}}'
FENCED = "```json\n" + VALID + "\n```"
# policy references an undefined address -> referential validation error
INVALID_REF = '{"fortigate_config": {"firewall_policies": [{"name": "p", "srcintf": ["port1"], "dstintf": ["port2"], "srcaddr": ["ghost"], "dstaddr": ["all"], "service": ["ALL"]}]}}'


def test_valid_json_builds_model():
    invoke = FakeInvoke([VALID])
    model, errors = asyncio.run(build_config_model(invoke, {"implementation_intent": {}}))
    assert isinstance(model, FortiGateConfigModel)
    assert model.hostname == "x"
    assert errors == []
    assert invoke.calls == 1


def test_fenced_json_parsed():
    invoke = FakeInvoke([FENCED])
    model, _ = asyncio.run(build_config_model(invoke, {}))
    assert isinstance(model, FortiGateConfigModel)


def test_retry_after_garbage_then_valid():
    invoke = FakeInvoke(["not json at all", VALID])
    model, errors = asyncio.run(build_config_model(invoke, {}, retries=1))
    assert isinstance(model, FortiGateConfigModel)
    assert invoke.calls == 2
    assert len(errors) == 1  # one failed attempt recorded


def test_retry_after_referential_error_then_valid():
    invoke = FakeInvoke([INVALID_REF, VALID])
    model, errors = asyncio.run(build_config_model(invoke, {}, retries=1))
    assert isinstance(model, FortiGateConfigModel)
    assert invoke.calls == 2
    assert any("ghost" in e for e in errors)


def test_persistent_failure_returns_none():
    invoke = FakeInvoke(["garbage", "still garbage"])
    model, errors = asyncio.run(build_config_model(invoke, {}, retries=1))
    assert model is None
    assert len(errors) == 2
    assert invoke.calls == 2


def test_version_normalization():
    assert normalize_fortios_version("7.4.5") == "7.4"
    assert normalize_fortios_version("v7.2") == "7.4"  # 7.2 not shipped yet -> default
    assert normalize_fortios_version("") == "7.4"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok", t.__name__)
    print(f"PASSED {len(TESTS)} config-builder tests")
