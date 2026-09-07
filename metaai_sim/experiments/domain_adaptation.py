"""Configuration and gating for domain-generalisation / adaptation runs.

Nothing is enabled by default. The core profile does **not** wire these
suites in; they live in ``experiments/profiles/domain_adaptation.yaml``.
The runner only executes them when:

* Tasks 1-3 have been completed (an explicit gesture set + inventory +
  same-room / different-date split are declared in the profile), AND
* the required source environments actually exist in the manifest.

Distinctions preserved here:

* **Domain generalisation (DG)** — no target-domain data used during
  training or model selection. Requires >=2 labelled source environments
  (rooms / dates) or the arm is marked ``unavailable`` with the exact
  reason "insufficient source environments".
* **Unsupervised adaptation (UDA)** — unlabelled target-domain features
  are used during training, target labels are never used. Also requires
  >=1 target environment with features but no labels released to the
  optimiser.

Every DA/DG evaluation is compared against the chance/majority floor AND
the in-domain ceiling (from ``raw_indomain`` runs on the same profile).
Prediction collapse is reported alongside accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class DomainArmSpec:
    """Registry entry for a DA/DG method."""

    name: str
    kind: str                     # "dg" | "uda"
    method: str                   # "erm" | "dann" | "coral" | "irm" | ...
    feature_mode: str = "raw"
    dfs_bins: str = "full"
    min_source_environments: int = 2      # DG minimum
    requires_unlabelled_target: bool = False
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    lambda_domain: float = 0.3            # DANN adversarial weight
    coral_weight: float = 1.0
    irm_penalty: float = 1.0
    extra: dict = field(default_factory=dict)


_DOMAIN_REGISTRY: dict[str, DomainArmSpec] = {}


def register(spec: DomainArmSpec) -> DomainArmSpec:
    if spec.name in _DOMAIN_REGISTRY:
        raise ValueError(f"domain arm {spec.name!r} already registered")
    _DOMAIN_REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> DomainArmSpec:
    if name not in _DOMAIN_REGISTRY:
        raise KeyError(
            f"unknown domain arm {name!r}. Registered: {sorted(_DOMAIN_REGISTRY)}"
        )
    return _DOMAIN_REGISTRY[name]


def registered_names() -> list[str]:
    return sorted(_DOMAIN_REGISTRY)


# Canonical DA/DG arms. All disabled from ``core`` profile — enabled only
# when ``profiles/domain_adaptation.yaml`` is selected AND gating passes.
register(DomainArmSpec("erm_raw", kind="dg", method="erm",
                        min_source_environments=1))
register(DomainArmSpec("dann_raw", kind="dg", method="dann",
                        min_source_environments=2))
register(DomainArmSpec("coral_raw", kind="dg", method="coral",
                        min_source_environments=2))
register(DomainArmSpec("irm_raw", kind="dg", method="irm",
                        min_source_environments=2))
register(DomainArmSpec("uda_dann_raw", kind="uda", method="dann",
                        min_source_environments=1,
                        requires_unlabelled_target=True))
register(DomainArmSpec("uda_coral_raw", kind="uda", method="coral",
                        min_source_environments=1,
                        requires_unlabelled_target=True))


@dataclass
class GateReport:
    ok: bool
    reason: str
    n_source_environments: int
    has_unlabelled_target: bool


def gate(
    spec: DomainArmSpec,
    *,
    source_environments: Sequence[str],
    has_unlabelled_target: bool,
    tasks_1_3_ready: bool,
) -> GateReport:
    """Decide whether ``spec`` may run.

    Refuses a multi-environment method when only one training environment
    exists — the ``reason`` string echoes the wording the task specification
    requires (``"insufficient source environments"``).
    """
    n_src = len(list(source_environments))
    if not tasks_1_3_ready:
        return GateReport(
            ok=False,
            reason=(
                "tasks 1-3 not complete: gesture set, duplicate-split "
                "resolution, and inventory must be recorded before "
                "domain-adaptation experiments run."
            ),
            n_source_environments=n_src,
            has_unlabelled_target=has_unlabelled_target,
        )
    if n_src < spec.min_source_environments:
        return GateReport(
            ok=False,
            reason=(
                "insufficient source environments "
                f"(have {n_src}, need >={spec.min_source_environments} for "
                f"method={spec.method!r})"
            ),
            n_source_environments=n_src,
            has_unlabelled_target=has_unlabelled_target,
        )
    if spec.requires_unlabelled_target and not has_unlabelled_target:
        return GateReport(
            ok=False,
            reason=(
                "unsupervised adaptation requires unlabelled target features; "
                "none were declared for this split"
            ),
            n_source_environments=n_src,
            has_unlabelled_target=has_unlabelled_target,
        )
    return GateReport(
        ok=True, reason="gated on: tasks_1_3_ready and source_environments",
        n_source_environments=n_src,
        has_unlabelled_target=has_unlabelled_target,
    )
