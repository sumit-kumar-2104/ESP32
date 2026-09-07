"""Detect configured splits that resolve to identical membership.

A ``Split`` is the tuple ``(sorted train_ids, sorted val_ids, sorted
test_ids)``. Two configured splits collide when those three sorted lists
are byte-identical after materialisation. Collisions are common when the
packaged Widar3.0 tree only exposes one date per room (e.g.
``cross_room_1_to_2`` and ``heldout_test_20181118`` become the same
partition because room 1 is only 20181109 and room 2 is only 20181118).

Canonical-vs-alias resolution:

- The FIRST split in profile order is the canonical id.
- Every colliding split is recorded as an *alias* — its status is
  ``alias_of``, its results are not counted, and the alias mapping is
  reported explicitly in the run manifest and RUN_SUMMARY.

Never present the same underlying evaluation as independent evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

from experiments.splits import Split


def split_fingerprint(split: Split) -> str:
    """Stable hash of ``(sorted train_ids, sorted val_ids, sorted test_ids)``.

    Grouping key and split kind are deliberately excluded — two different
    configurations that end up on the same rows ARE the same evaluation.
    """
    h = hashlib.sha256()
    for name, ids in (("train", split.train_ids),
                       ("val", split.val_ids),
                       ("test", split.test_ids)):
        h.update(name.encode("utf-8"))
        h.update(b":")
        for s in sorted(ids):
            h.update(s.encode("utf-8"))
            h.update(b",")
        h.update(b"|")
    return h.hexdigest()[:16]


@dataclass
class DedupResult:
    canonical: list[tuple[str, Split]]                 # (split_id, split)
    aliases: dict[str, str] = field(default_factory=dict)   # alias_id -> canonical_id
    fingerprints: dict[str, str] = field(default_factory=dict)  # split_id -> fingerprint

    @property
    def n_unique(self) -> int:
        return len(self.canonical)

    @property
    def n_aliases(self) -> int:
        return len(self.aliases)

    def to_manifest_dict(self) -> dict:
        return {
            "canonical_split_ids": [sid for sid, _ in self.canonical],
            "aliases": dict(self.aliases),
            "fingerprints": dict(self.fingerprints),
            "n_unique": self.n_unique,
            "n_aliases": self.n_aliases,
        }


def dedup_splits(
    concrete: Sequence[tuple[str, Split]],
) -> DedupResult:
    """Return the canonical split list and the alias mapping."""
    canonical: list[tuple[str, Split]] = []
    fp_to_canonical: dict[str, str] = {}
    aliases: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    for split_id, split in concrete:
        fp = split_fingerprint(split)
        fingerprints[split_id] = fp
        if fp in fp_to_canonical:
            aliases[split_id] = fp_to_canonical[fp]
        else:
            fp_to_canonical[fp] = split_id
            canonical.append((split_id, split))
    return DedupResult(
        canonical=canonical, aliases=aliases, fingerprints=fingerprints,
    )
