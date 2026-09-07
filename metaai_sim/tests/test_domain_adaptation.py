"""Domain-adaptation gating: refuse multi-env method with single env,
refuse UDA without unlabelled target, refuse everything until tasks 1-3."""

from __future__ import annotations

import pytest

from experiments import domain_adaptation as da


def test_refuse_when_tasks_1_3_not_ready() -> None:
    spec = da.get("erm_raw")
    report = da.gate(
        spec, source_environments=["room1_date20181109"],
        has_unlabelled_target=False, tasks_1_3_ready=False,
    )
    assert report.ok is False
    assert "tasks 1-3" in report.reason


def test_refuse_multi_env_method_with_single_environment() -> None:
    spec = da.get("dann_raw")
    report = da.gate(
        spec, source_environments=["room1_date20181109"],
        has_unlabelled_target=False, tasks_1_3_ready=True,
    )
    assert report.ok is False
    assert "insufficient source environments" in report.reason


def test_allow_multi_env_method_with_two_environments() -> None:
    spec = da.get("dann_raw")
    report = da.gate(
        spec,
        source_environments=["room1_date20181109", "room1_date20181115"],
        has_unlabelled_target=False, tasks_1_3_ready=True,
    )
    assert report.ok is True


def test_refuse_uda_without_unlabelled_target() -> None:
    spec = da.get("uda_dann_raw")
    report = da.gate(
        spec, source_environments=["room1_date20181109"],
        has_unlabelled_target=False, tasks_1_3_ready=True,
    )
    assert report.ok is False
    assert "unsupervised adaptation" in report.reason.lower()


def test_allow_uda_with_unlabelled_target() -> None:
    spec = da.get("uda_dann_raw")
    report = da.gate(
        spec, source_environments=["room1_date20181109"],
        has_unlabelled_target=True, tasks_1_3_ready=True,
    )
    assert report.ok is True


def test_erm_allowed_with_single_environment() -> None:
    """ERM is defined as ``min_source_environments=1`` so it must be
    allowed with a single labelled source. Distinguishes DG-with-erm
    (baseline) from multi-env DG."""
    spec = da.get("erm_raw")
    report = da.gate(
        spec, source_environments=["room1_date20181109"],
        has_unlabelled_target=False, tasks_1_3_ready=True,
    )
    assert report.ok is True


def test_registered_names_include_core_dg_and_uda_methods() -> None:
    names = set(da.registered_names())
    assert {"erm_raw", "dann_raw", "coral_raw", "irm_raw",
            "uda_dann_raw", "uda_coral_raw"}.issubset(names)


def test_unknown_arm_lookup_raises() -> None:
    with pytest.raises(KeyError):
        da.get("nonexistent_arm")
