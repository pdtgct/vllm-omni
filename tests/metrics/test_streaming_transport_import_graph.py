# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-5 tests-first: the streaming_transport import-graph guarantee.

Specs: PORT-OBS-003's "Observer protocol home" decision — the observer
protocol and handle type live in a neutral, Prometheus-free module so the
model package can import it and the GPU worker/scheduler can call its
batch-stat transport hops without ever touching Prometheus.

DISCOVERY (documented rather than silently worked around): a bare
``import vllm_omni`` already puts ``prometheus_client`` in ``sys.modules``
in this checkout, via the top-level package's own eager imports —
entirely unrelated to this module. A raw subprocess "import
vllm_omni.metrics.streaming_transport, then check sys.modules delta for
prometheus_client" check therefore fails today regardless of anything
this module does, because Python always executes a package's
``__init__.py`` before any of its submodules. That is a pre-existing,
separate fact about ``vllm_omni``'s own lazy-import discipline — out of
scope for this Phase-5 pass to fix. What IS meaningfully testable and
enforced here, matching the actual "worker-safe" claim:

1. streaming_transport.py's OWN source imports neither ``prometheus_client``
   nor ``vllm_omni.metrics.streaming`` (source-scan — a real, enforced
   control on this module's authors, real GREEN today).
2. Loading streaming_transport.py IN ISOLATION (stubbed parent packages,
   loaded by file path — the same pattern this file's sibling tests use to
   avoid vllm_omni's own package-init cost) pulls in neither
   ``prometheus_client`` nor any Prometheus family registration — the
   worker-process-safe claim this module actually needs to satisfy,
   verified in a subprocess.
3. No ``vllm_omni:streaming_*`` family registers as a side effect.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TRANSPORT_PATH = Path(__file__).resolve().parents[2] / "vllm_omni/metrics/streaming_transport.py"


# ---- source-scan: no prometheus_client / streaming.py import (real, GREEN) ---


# @spec PORT-OBS-003
def test_streaming_transport_source_imports_neither_prometheus_nor_streaming() -> None:
    import re

    text = _TRANSPORT_PATH.read_text()
    banned = re.compile(
        r"^\s*(import prometheus_client|from prometheus_client"
        r"|from vllm_omni\.metrics\.streaming import"
        r"|from vllm_omni\.metrics import streaming\b)",
        re.MULTILINE,
    )
    assert banned.search(text) is None, "streaming_transport.py must stay Prometheus- and streaming.py-free"


# ---- isolated-load worker-safety (real, GREEN) --------------------------------


# @spec PORT-OBS-003
def test_isolated_load_pulls_in_neither_prometheus_client_nor_streaming_family() -> None:
    """Mirrors the stub-parent-package + load-by-file-path pattern every
    other loader-chain test in this suite already uses (the actual
    "worker process never needs Prometheus" claim), run in a subprocess
    so the sys.modules delta is clean."""
    code = textwrap.dedent(
        f"""
        import importlib.util
        import sys
        import types

        for name in ("vllm_omni", "vllm_omni.metrics"):
            sys.modules[name] = types.ModuleType(name)

        before = set(sys.modules)
        spec = importlib.util.spec_from_file_location(
            "vllm_omni.metrics.streaming_transport", {str(_TRANSPORT_PATH)!r}
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        after = set(sys.modules)

        delta = after - before
        print("PROMETHEUS_IN_DELTA", "prometheus_client" in delta)
        print("HAS_CHUNK_READY_HANDLE", hasattr(module, "ChunkReadyHandle"))
        print("HAS_STREAMING_OBSERVER", hasattr(module, "StreamingObserver"))
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "PROMETHEUS_IN_DELTA False" in result.stdout, result.stdout
    assert "HAS_CHUNK_READY_HANDLE True" in result.stdout
    assert "HAS_STREAMING_OBSERVER True" in result.stdout


# @spec PORT-OBS-001, PORT-OBS-003
def test_isolated_load_registers_no_streaming_prometheus_family() -> None:
    """Importing streaming_transport alone must register zero
    ``vllm_omni:streaming_*`` families — those live only in
    ``vllm_omni.metrics.streaming``, never loaded here."""
    code = textwrap.dedent(
        f"""
        import importlib.util
        import sys
        import types

        for name in ("vllm_omni", "vllm_omni.metrics"):
            sys.modules[name] = types.ModuleType(name)

        spec = importlib.util.spec_from_file_location(
            "vllm_omni.metrics.streaming_transport", {str(_TRANSPORT_PATH)!r}
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        assert "prometheus_client" not in sys.modules
        print("OK: no prometheus_client, so no registry to scrape either")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


# ---- documents the pre-existing, unrelated fact (real, GREEN) ----------------


# @spec PORT-OBS-003
def test_bare_vllm_omni_import_already_pulls_in_prometheus_client() -> None:
    """DISCOVERY pin, not a regression this pass introduces: a bare
    ``import vllm_omni`` already puts prometheus_client in sys.modules
    today, from the top-level package's own eager imports — unrelated to
    streaming work. Documents why a literal "subprocess import
    vllm_omni.metrics.streaming_transport, check sys.modules" check is
    not a meaningful test of THIS module in isolation."""
    code = "import sys\nimport vllm_omni\nprint('prometheus_client' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "True" in result.stdout
