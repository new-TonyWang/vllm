# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GPUWorker weight-transfer pass-through behavior.

The worker no longer contains transport, layerwise, or sparse logic: it only
delegates to the configured weight transfer engine and tracks whether an update
session is active. These tests verify that delegation and the session guard.
"""

import pytest
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.v1.worker import gpu_worker as gpu_worker_module
from vllm.v1.worker.gpu_model_runner import _get_parameter_for_reload
from vllm.v1.worker.gpu_worker import Worker


class _RecordingEngine:
    """Minimal stand-in for a weight transfer engine."""

    def __init__(
        self, raise_on_update: bool = False, raise_on_finish: bool = False
    ) -> None:
        self.raise_on_update = raise_on_update
        self.raise_on_finish = raise_on_finish
        self.started = False
        self.finished = False
        self.reset_count = 0
        self.supports_draft_weight_update = False
        self.update_calls: list[dict] = []
        self.seen_configs: list[VllmConfig] = []

    def _record_config(self) -> None:
        self.seen_configs.append(get_current_vllm_config())

    def start_weight_update(self) -> None:
        self._record_config()
        self.started = True

    def update_weights(self, update_info: dict) -> None:
        self._record_config()
        self.update_calls.append(update_info)
        if self.raise_on_update:
            raise ValueError("boom")

    def finish_weight_update(self) -> None:
        self._record_config()
        if self.raise_on_finish:
            raise ValueError("finish boom")
        self.finished = True

    def reset_weight_update_target(self) -> None:
        self.reset_count += 1


class _RecordingModelRunner:
    def __init__(self) -> None:
        self.seen_config: VllmConfig | None = None
        self.reset_lora_calls = 0

    def reload_weights(self) -> None:
        self.seen_config = get_current_vllm_config()

    def reset_lora_state(self) -> None:
        self.reset_lora_calls += 1


def _make_worker(engine: _RecordingEngine | None) -> Worker:
    worker = object.__new__(Worker)
    worker.vllm_config = VllmConfig()
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False
    worker._weight_update_is_draft = False
    worker._weight_update_failed = False
    worker._weight_update_generation_id = None
    worker._weight_update_expected_names = None
    worker._weight_update_received_names = set()
    worker._weight_update_allow_partial = False
    worker.model_runner = _RecordingModelRunner()
    return worker


def test_reload_weights_sets_current_config():
    worker = _make_worker(None)
    model_runner = _RecordingModelRunner()
    worker.model_runner = model_runner  # type: ignore[assignment]

    Worker.reload_weights(worker)

    assert model_runner.seen_config is worker.vllm_config


def test_reload_parameter_lookup_preserves_lora_module_names():
    base_layer = nn.Module()
    qweight = nn.Parameter(torch.ones(1))
    base_layer.register_parameter("qweight", qweight)
    wrapper = BaseLayerWithLoRA()
    wrapper.base_layer = base_layer
    model = nn.Module()
    model.proj = wrapper

    named_parameters = dict(model.named_parameters())
    assert set(named_parameters) == {"proj.base_layer.qweight"}
    assert named_parameters["proj.base_layer.qweight"] is qweight
    assert model.get_parameter("proj.base_layer.qweight") is qweight
    assert _get_parameter_for_reload(model, "proj.qweight") is qweight


def test_start_update_finish_delegates_to_engine():
    engine = _RecordingEngine()
    worker = _make_worker(engine)

    Worker.start_weight_update(worker)
    assert engine.started is True
    assert worker._weight_update_active is True
    with pytest.raises(RuntimeError, match="transaction is active"):
        Worker.check_health(worker)

    Worker.update_weights(worker, {"names": ["w"]})
    assert engine.update_calls == [{"names": ["w"]}]
    assert worker._weight_update_active is True

    Worker.finish_weight_update(worker)
    assert engine.finished is True
    assert engine.reset_count == 1
    assert worker._weight_update_active is False
    assert engine.seen_configs == [worker.vllm_config] * 3
    assert worker.model_runner.reset_lora_calls == 1
    Worker.check_health(worker)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(1, {"names": ["rank-1"]}), (2, {"names": []})],
)
def test_rank_local_update_selects_worker_payload(rank, expected):
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    worker.rank = rank
    Worker.start_weight_update(worker)

    Worker.update_weights(
        worker, [{"names": ["rank-0"]}, {"names": ["rank-1"]}, {"names": []}]
    )

    assert engine.update_calls == [expected]
    assert worker._weight_update_active is True


def test_rank_local_update_includes_data_parallel_rank():
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    worker.rank = 0
    worker.vllm_config.parallel_config.data_parallel_size = 4
    worker.vllm_config.parallel_config.data_parallel_rank = 2
    Worker.start_weight_update(worker)

    Worker.update_weights(
        worker,
        [
            {"names": ["dp-0"]},
            {"names": ["dp-1"]},
            {"names": ["dp-2"]},
            {"names": ["dp-3"]},
        ],
    )

    assert engine.update_calls == [{"names": ["dp-2"]}]
    assert worker._weight_update_active is True


def test_finish_draft_session_keeps_lora_state():
    engine = _RecordingEngine()
    engine.supports_draft_weight_update = True
    worker = _make_worker(engine)
    worker._set_draft_weight_update_target = lambda: None

    Worker.start_draft_weight_update(worker)
    Worker.finish_weight_update(worker)

    assert worker.model_runner.reset_lora_calls == 0


def test_double_start_raises():
    worker = _make_worker(_RecordingEngine())
    Worker.start_weight_update(worker)
    with pytest.raises(RuntimeError, match="already"):
        Worker.start_weight_update(worker)


def test_update_without_start_raises():
    worker = _make_worker(_RecordingEngine())
    with pytest.raises(RuntimeError, match="start_weight_update must be called"):
        Worker.update_weights(worker, {"names": ["w"]})


def test_finish_without_start_raises():
    worker = _make_worker(_RecordingEngine())
    with pytest.raises(RuntimeError, match="without a matching"):
        Worker.finish_weight_update(worker)


def test_update_failure_disables_serving_until_full_reload():
    engine = _RecordingEngine(raise_on_update=True)
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)

    with pytest.raises(ValueError, match="boom"):
        Worker.update_weights(worker, {"names": ["w"]})

    assert engine.reset_count == 1
    assert worker._weight_update_active is False
    with pytest.raises(RuntimeError, match="mixed generations"):
        Worker.check_health(worker)
    with pytest.raises(RuntimeError, match="mixed generations"):
        Worker.start_weight_update(worker)

    Worker.reload_weights(worker)
    Worker.check_health(worker)


def test_finish_failure_disables_serving_and_ends_session():
    engine = _RecordingEngine(raise_on_finish=True)
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)

    with pytest.raises(ValueError, match="finish boom"):
        Worker.finish_weight_update(worker)

    assert engine.reset_count == 1
    assert worker._weight_update_active is False
    assert worker.model_runner.reset_lora_calls == 0
    with pytest.raises(RuntimeError, match="mixed generations"):
        Worker.check_health(worker)


def test_manifest_rejects_missing_name_at_finish():
    worker = _make_worker(_RecordingEngine())
    Worker.start_weight_update(
        worker,
        manifest={
            "generation_id": "generation-1",
            "expected_parameter_names": ["a", "b"],
        },
    )
    Worker.update_weights(worker, {"names": ["a"]})

    with pytest.raises(ValueError, match="missing names"):
        Worker.finish_weight_update(worker, generation_id="generation-1")

    with pytest.raises(RuntimeError, match="mixed generations"):
        Worker.check_health(worker)


def test_manifest_rejects_unknown_name_before_engine_update():
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    Worker.start_weight_update(
        worker,
        manifest={
            "generation_id": "generation-1",
            "expected_parameter_names": ["expected"],
        },
    )

    with pytest.raises(ValueError, match="absent from the START manifest"):
        Worker.update_weights(worker, {"names": ["unknown"]})

    assert engine.update_calls == []


def test_manifest_rejects_generation_mismatch():
    worker = _make_worker(_RecordingEngine())
    Worker.start_weight_update(
        worker,
        manifest={
            "generation_id": "generation-1",
            "expected_parameter_names": [],
        },
    )

    with pytest.raises(ValueError, match="generation mismatch"):
        Worker.finish_weight_update(worker, generation_id="generation-2")


def test_manifest_rejects_missing_finish_generation():
    worker = _make_worker(_RecordingEngine())
    Worker.start_weight_update(
        worker,
        manifest={
            "generation_id": "generation-1",
            "expected_parameter_names": [],
        },
    )

    with pytest.raises(ValueError, match="generation mismatch"):
        Worker.finish_weight_update(worker)

    with pytest.raises(RuntimeError, match="mixed generations"):
        Worker.check_health(worker)


def test_legacy_start_does_not_require_finish_generation():
    worker = _make_worker(_RecordingEngine())

    Worker.start_weight_update(worker)
    Worker.finish_weight_update(worker)

    Worker.check_health(worker)


def test_manifest_allows_declared_partial_update():
    worker = _make_worker(_RecordingEngine())
    Worker.start_weight_update(
        worker,
        manifest={
            "generation_id": "generation-1",
            "expected_parameter_names": ["a", "b"],
            "allow_partial": True,
        },
    )
    Worker.update_weights(worker, {"names": ["a"]})
    Worker.finish_weight_update(worker, generation_id="generation-1")

    Worker.check_health(worker)


def test_remote_rank_metadata_failure_stops_all_ranks_before_receive(monkeypatch):
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)
    worker.vllm_config.parallel_config.world_size = 2
    cpu_group = object()
    world = type(
        "FakeWorld",
        (),
        {"world_size": 2, "ranks": [0, 1], "cpu_group": cpu_group},
    )()
    monkeypatch.setattr(gpu_worker_module, "get_world_group", lambda: world)

    def gather_failures(failures, local_failure, *, group):
        assert group is cpu_group
        assert local_failure is None
        failures[:] = [None, "ValueError: unknown parameter"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_failures)

    with pytest.raises(RuntimeError, match="rank 1: ValueError"):
        Worker.update_weights(worker, {"names": ["expected"]})

    assert engine.update_calls == []
    with pytest.raises(RuntimeError, match="mixed generations"):
        Worker.check_health(worker)


def test_remote_rank_finish_failure_latches_successful_rank(monkeypatch):
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    Worker.start_weight_update(
        worker,
        manifest={
            "generation_id": "generation-1",
            "expected_parameter_names": [],
        },
    )
    worker.vllm_config.parallel_config.world_size = 2
    cpu_group = object()
    world = type(
        "FakeWorld",
        (),
        {"world_size": 2, "ranks": [0, 1], "cpu_group": cpu_group},
    )()
    monkeypatch.setattr(gpu_worker_module, "get_world_group", lambda: world)
    gather_count = 0

    def gather_failures(failures, local_failure, *, group):
        nonlocal gather_count
        assert group is cpu_group
        assert local_failure is None
        gather_count += 1
        failures[:] = (
            [None, None] if gather_count == 1 else [None, "ValueError: finish boom"]
        )

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_failures)

    with pytest.raises(RuntimeError, match="rank 1: ValueError"):
        Worker.finish_weight_update(worker, generation_id="generation-1")

    assert engine.finished is True
    assert worker.model_runner.reset_lora_calls == 0
    with pytest.raises(RuntimeError, match="mixed generations"):
        Worker.check_health(worker)


def test_missing_engine_raises():
    worker = _make_worker(None)
    with pytest.raises(RuntimeError, match="Weight transfer not configured"):
        Worker.start_weight_update(worker)
