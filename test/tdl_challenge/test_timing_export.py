"""Tests for challenge runtime and parameter export."""

import importlib

import torch.nn as nn

from topobench.callbacks.timer_callback import PipelineTimer

challenge_utils = importlib.import_module("2026_tdl_challenge.utils")


def test_collect_run_metadata_exports_parameters_and_epoch_times():
    """Export parameter counts and post-warm-up epoch statistics."""
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.Linear(4, 2, bias=False),
    )
    model[1].weight.requires_grad_(False)
    timer = PipelineTimer()
    timer.sums["train_epoch"] = list(range(1, 13))

    metadata = challenge_utils._collect_run_metadata(
        {"model": model, "callbacks": [timer]}
    )

    assert metadata["model_params_total"] == 24
    assert metadata["model_params_trainable"] == 16
    assert metadata["model_params_non_trainable"] == 8
    assert metadata["train_epochs_total"] == 12
    assert metadata["train_epoch_warmup_epochs_excluded"] == 10
    assert metadata["train_epochs_timed"] == 2
    assert metadata["train_epoch_time_mean_seconds"] == 11.5
    assert metadata["train_epoch_time_std_seconds"] == 0.5


def test_collect_run_metadata_retains_one_short_run_measurement():
    """Retain one timing sample when a run is shorter than the warm-up."""
    model = nn.Linear(2, 1)
    timer = PipelineTimer()
    timer.sums["train_epoch"] = [2.5]

    metadata = challenge_utils._collect_run_metadata(
        {"model": model, "callbacks": [timer]}
    )

    assert metadata["train_epochs_total"] == 1
    assert metadata["train_epoch_warmup_epochs_excluded"] == 0
    assert metadata["train_epochs_timed"] == 1
    assert metadata["train_epoch_time_mean_seconds"] == 2.5
    assert metadata["train_epoch_time_std_seconds"] == 0.0


def test_collect_run_metadata_without_timer_exports_only_parameters():
    """Export parameter counts when no pipeline timer is configured."""
    model = nn.Linear(2, 1)

    metadata = challenge_utils._collect_run_metadata(
        {"model": model, "callbacks": []}
    )

    assert metadata == {
        "model_params_total": 3,
        "model_params_trainable": 3,
        "model_params_non_trainable": 0,
    }
