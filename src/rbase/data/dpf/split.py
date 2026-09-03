# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Group split: each DPF/protein/sequence family is train XOR val XOR test.

The split atom is a *biological identity component*, not a raw sequence string.
Two families are forced onto the same side of the split when they

1. share an on-disk structure (same XTC / topology PDB / ATLAS family dir),
2. have byte-identical ``seqres``,
3. come from the same PDB entry (``1abc_A`` and ``1abc_B``), or
4. are sequence homologs: one ``seqres`` contains the other, or their k-mer
   containment exceeds :data:`KMER_CONTAINMENT_THRESHOLD`.

Rule 4 is what makes the split an experiment rather than a formality: without it
two chains of the same protein that differ by a couple of resolved residues land
on opposite sides while the leak check passes clean.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from typing import Iterable, Sequence, Union

from .catalog import DpfCatalog, DpfFamily

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

SPLIT_NAMES = ("train", "val", "test")

SPLIT_PAYLOAD_VERSION = 3

# --- Homology grouping constants -------------------------------------------
# k-mer length used to compare sequences without an external aligner
# (no mmseqs/CD-HIT dependency is allowed here).
KMER_SIZE = 8
# Containment = |kmers(A) & kmers(B)| / min(|kmers(A)|, |kmers(B)|).
# For two sequences of identity p, the expected fraction of shared k-mers is
# roughly p**KMER_SIZE, so 0.5 at k=8 corresponds to ~92% sequence identity.
# Trade-off: lower it and unrelated-but-similar folds get merged, shrinking the
# usable holdout; raise it and true near-duplicates (single-point mutants,
# differently resolved copies of the same chain) leak across the split. Erring
# low only costs holdout size, whereas erring high silently invalidates the
# evaluation -- but on the full 1938-chain ATLAS tree, scored against the RCSB
# 95%-identity clusters shipped with the dataset, dropping 0.5 to 0.2 recovers
# only one extra homolog pair (306 -> 307 of 309 candidate pairs) while merging
# 15 more pairs that are not in the same 95% cluster. 0.5 is that knee.
KMER_CONTAINMENT_THRESHOLD = 0.5
# Substring/containment rule: a sequence fully contained in another one is the
# same protein with different resolved termini. Short peptides are excluded so
# that toy/fragment sequences do not chain unrelated families together.
MIN_SUBSTRING_LENGTH = 20
# k-mers shared by more than this fraction of families carry no identity signal
# (low-complexity runs); they are skipped when generating candidate pairs so the
# grouping stays O(n^2)-safe for ~2000 families.
MAX_KMER_FAMILY_FRACTION = 0.25
# ...but a pure fraction turns the homology rule OFF on small catalogs: with 8
# families the cap is 2, so a k-mer shared by a 4-family cluster of point
# mutants is discarded and the cluster silently stops being one component --
# exactly the case ``assert_no_leakage`` is supposed to catch. The cap only
# exists to bound work, so give it an absolute floor, and below
# ``KMER_INDEX_ALL_PAIRS_MAX_FAMILIES`` do not cap at all (n^2 is trivial there).
MIN_KMER_POSTINGS = 50
KMER_INDEX_ALL_PAIRS_MAX_FAMILIES = 500

# Rule 3 (same PDB entry) keys off the part of the family id before the first
# ``_``. That is only meaningful when it actually looks like a PDB entry id and
# the remainder looks like a chain id: without the shape check a catalog of
# ``dpf_000 … dpf_011`` collapses into a single component.
PDB_ENTRY_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
PDB_CHAIN_ID_RE = re.compile(r"^[A-Za-z0-9]{1,4}$")

class StructuralLeakError(ValueError):
    """Raised when a family, sequence, or member would cross splits."""

class SplitConfigMismatchError(ValueError):
    """Raised when a persisted split disagrees with the requested split config."""

@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self):
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1, got {total}")
        if min(self.train, self.val, self.test) < 0:
            raise ValueError("Split fractions must be non-negative")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in SPLIT_NAMES}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "SplitFractions":
        return cls(
            train=float(payload["train"]),
            val=float(payload["val"]),
            test=float(payload["test"]),
        )

    def positive_names(self) -> list[str]:
        return [name for name in SPLIT_NAMES if getattr(self, name) > 0]

    def approx_equal(self, other: "SplitFractions", tol: float = 1e-6) -> bool:
        return all(
            abs(getattr(self, name) - getattr(other, name)) <= tol
            for name in SPLIT_NAMES
        )

    def assign(self, unit: float) -> str:
        """Map a point in [0, 1) onto a split name.

        Legacy helper. ``DpfSplit.from_catalog`` no longer hashes components
        independently into this function -- that drifted badly from the
        requested fractions and could produce an empty val/test -- it uses the
        quota assignment in :func:`quota_assignment` instead.
        """
        if unit < self.train:
            return "train"
        if unit < self.train + self.val:
            return "val"
        return "test"

POLICY_COUNTS = "counts"
POLICY_FRACTIONS = "fractions"
SPLIT_POLICIES = (POLICY_COUNTS, POLICY_FRACTIONS)

@dataclass
class DpfSplit:
    """family_id → {'train','val','test'}. Members are never assigned independently."""

    seed: int
    assignment: dict[str, str]
    fractions: SplitFractions | None = None
    catalog_fingerprint: str | None = None
    # Which policy produced this split. ``'counts'`` is the DPF default
    # (--n_holdout / --n_train); ``'fractions'`` is --frac_split. Persisted so a
    # stored split cannot be silently reused under a different policy or a
    # different --n_holdout -- ``fractions`` alone cannot express that, because
    # the count policy does not have any.
    policy: str | None = None
    n_holdout: int | None = None
    n_train: int | None = None
    #: Families carved out for validation under the count policy. Persisted for
    #: the same reason as n_holdout: without it, changing --n_val silently reuses
    #: a split that has no val families, and the conflict only surfaces later as
    #: a confusing "zero val families" error from assert_split_populated.
    n_val: int | None = None

    def families(self, split_name: str) -> list[str]:
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {split_name!r}; expected {SPLIT_NAMES}")
        return sorted(
            fid for fid, name in self.assignment.items() if name == split_name
        )

    def counts(self) -> dict[str, int]:
        return {name: len(self.families(name)) for name in SPLIT_NAMES}

    def save(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SPLIT_PAYLOAD_VERSION,
            "seed": self.seed,
            "policy": self.policy,
            "fractions": None if self.fractions is None else self.fractions.as_dict(),
            "catalog_fingerprint": self.catalog_fingerprint,
            "assignment": dict(sorted(self.assignment.items())),
        }
        if self.policy == POLICY_COUNTS:
            payload["n_holdout"] = self.n_holdout
            payload["n_train"] = self.n_train
            payload["n_val"] = self.n_val
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(
        cls,
        path: PathLike,
        catalog: DpfCatalog | None = None,
        expect_seed: int | None = None,
        expect_fractions: SplitFractions | None = None,
        expect_n_holdout: int | None = None,
        expect_n_train: int | None = None,
        expect_n_val: int | None = None,
        expect_policy: str | None = None,
    ) -> "DpfSplit":
        """Load a persisted split and refuse to silently override the request.

        A stored split wins over the command line only when it *matches* it.
        ``expect_seed`` / ``expect_fractions`` / ``expect_n_holdout`` /
        ``expect_n_train`` are the values the caller asked for; when the catalog
        is given, its fingerprint is checked too. Any disagreement raises
        :class:`SplitConfigMismatchError`. Splits written before the
        fingerprint/policy fields existed are accepted with a warning.

        The requested policy is inferred from the expectations that were given
        (any count ⇒ ``'counts'``, else fractions ⇒ ``'fractions'``) unless
        ``expect_policy`` says otherwise, so the default DPF count policy is
        checked with the same strictness as ``--frac_split``.
        """
        path = Path(path)
        payload = json.loads(path.read_text())
        stored_fractions = payload.get("fractions")
        stored_policy = payload.get("policy")
        if stored_policy is not None and stored_policy not in SPLIT_POLICIES:
            raise SplitConfigMismatchError(
                f"Persisted split {path} records an unknown policy "
                f"{stored_policy!r}; expected one of {SPLIT_POLICIES}."
            )
        split = cls(
            seed=int(payload["seed"]),
            assignment=dict(payload["assignment"]),
            fractions=(
                None if stored_fractions is None
                else SplitFractions.from_dict(stored_fractions)
            ),
            catalog_fingerprint=payload.get("catalog_fingerprint"),
            policy=stored_policy,
            n_holdout=(
                None if payload.get("n_holdout") is None
                else int(payload["n_holdout"])
            ),
            n_train=(
                None if payload.get("n_train") is None else int(payload["n_train"])
            ),
            n_val=(
                None if payload.get("n_val") is None else int(payload["n_val"])
            ),
        )
        hint = (
            f"Pass --resplit to rebuild {path}, or point --split_file at a new path."
        )
        if expect_policy is not None and expect_policy not in SPLIT_POLICIES:
            raise ValueError(
                f"expect_policy={expect_policy!r}; expected one of {SPLIT_POLICIES}"
            )
        requested_policy = expect_policy
        if requested_policy is None:
            if expect_n_holdout is not None or expect_n_train is not None:
                requested_policy = POLICY_COUNTS
            elif expect_fractions is not None:
                requested_policy = POLICY_FRACTIONS
        if (
            expect_seed is None
            and requested_policy is None
            and expect_fractions is None
        ):
            logger.warning(
                f"Reusing the persisted split {path} without checking it against "
                f"any requested split config (no seed/policy/count expectation "
                f"was passed to DpfSplit.load); a changed --split_seed, "
                f"--n_holdout or --frac_split will be silently ignored."
            )
        if expect_seed is not None and int(expect_seed) != split.seed:
            raise SplitConfigMismatchError(
                f"Persisted split {path} was built with seed={split.seed}, but "
                f"seed={int(expect_seed)} was requested. {hint}"
            )
        if requested_policy is not None:
            if split.policy is None:
                logger.warning(
                    f"Persisted split {path} does not record its split policy "
                    f"(written by an older version); cannot verify it against the "
                    f"requested {requested_policy!r} policy."
                )
            elif split.policy != requested_policy:
                raise SplitConfigMismatchError(
                    f"Persisted split {path} was built with the "
                    f"{split.policy!r} split policy, but the {requested_policy!r} "
                    f"policy was requested. {hint}"
                )
        if requested_policy == POLICY_COUNTS and split.policy == POLICY_COUNTS:
            for label, expected, stored in (
                ("n_holdout", expect_n_holdout, split.n_holdout),
                ("n_train", expect_n_train, split.n_train),
                ("n_val", expect_n_val, split.n_val),
            ):
                if expected is None:
                    continue
                if stored is None:
                    logger.warning(
                        f"Persisted split {path} does not record {label} "
                        f"(written by an older version); cannot verify it against "
                        f"the requested {label}={int(expected)}."
                    )
                elif int(stored) != int(expected):
                    raise SplitConfigMismatchError(
                        f"Persisted split {path} was built with {label}={stored}, "
                        f"but {label}={int(expected)} was requested. {hint}"
                    )
        if expect_fractions is not None and requested_policy != POLICY_COUNTS:
            if split.fractions is None:
                logger.warning(
                    f"Persisted split {path} does not record its split fractions "
                    f"(written by an older version); cannot verify it against the "
                    f"requested {expect_fractions.as_dict()}."
                )
            elif not split.fractions.approx_equal(expect_fractions):
                raise SplitConfigMismatchError(
                    f"Persisted split {path} was built with fractions "
                    f"{split.fractions.as_dict()}, but "
                    f"{expect_fractions.as_dict()} was requested. {hint}"
                )
        if catalog is not None:
            fingerprint = catalog_fingerprint(catalog)
            if split.catalog_fingerprint is None:
                logger.warning(
                    f"Persisted split {path} does not record a catalog fingerprint "
                    f"(written by an older version); cannot verify that it was built "
                    f"from this catalog."
                )
            elif split.catalog_fingerprint != fingerprint:
                raise SplitConfigMismatchError(
                    f"Persisted split {path} was built from a different catalog "
                    f"(fingerprint {split.catalog_fingerprint[:12]}… vs "
                    f"{fingerprint[:12]}… for the catalog now loaded). {hint}"
                )
            assert_no_leakage(catalog, split)
        return split

    @classmethod
    def from_catalog(
        cls,
        catalog: DpfCatalog,
        seed: int,
        fractions: SplitFractions | None = None,
        n_holdout: int | None = None,
        n_train: int | None = None,
    ) -> "DpfSplit":
        if n_holdout is not None or n_train is not None:
            return cls._from_family_counts(
                catalog, seed=seed, n_holdout=n_holdout, n_train=n_train
            )
        fractions = fractions or SplitFractions()
        components = identity_components(catalog)
        groups = _groups_by_root(components)
        component_assignment = quota_assignment(groups, fractions=fractions, seed=seed)

        assignment = {
            family.family_id: component_assignment[components[family.family_id]]
            for family in catalog.families
        }
        split = cls(
            seed=seed,
            assignment=assignment,
            fractions=fractions,
            catalog_fingerprint=catalog_fingerprint(catalog),
            policy=POLICY_FRACTIONS,
        )
        assert_no_leakage(catalog, split)
        return split

    @classmethod
    def _from_family_counts(
        cls,
        catalog: DpfCatalog,
        seed: int,
        n_holdout: int | None,
        n_train: int | None,
    ) -> "DpfSplit":
        """Exact family counts. Whole identity components stay together.

        Default DPF policy: 90 train / 10 holdout (test), no val.
        """
        n_families = len(catalog.families)
        if n_holdout is None:
            if n_train is None:
                raise ValueError("n_holdout or n_train is required")
            n_holdout = n_families - int(n_train)
        n_holdout = int(n_holdout)
        if n_holdout < 0 or n_holdout > n_families:
            raise ValueError(
                f"n_holdout={n_holdout} is outside [0, {n_families}]"
            )
        if n_train is not None and int(n_train) + n_holdout != n_families:
            raise ValueError(
                f"n_train ({n_train}) + n_holdout ({n_holdout}) must equal "
                f"{n_families} families (val is unused in the count split)"
            )

        components = identity_components(catalog)
        groups = _groups_by_root(components)

        ranked_roots = _shuffled_roots(seed, groups)
        test_ids: list[str] = []
        for root in reversed(ranked_roots):
            fams = groups[root]
            if len(test_ids) + len(fams) <= n_holdout:
                test_ids.extend(fams)
            if len(test_ids) == n_holdout:
                break
        if len(test_ids) != n_holdout:
            raise StructuralLeakError(
                f"Cannot hold out exactly {n_holdout} families without splitting "
                f"a sequence/structure component (assigned {len(test_ids)}). "
                "Merge or pick a holdout count that is a sum of component sizes."
            )
        test_set = set(test_ids)
        assignment = {
            family.family_id: ("test" if family.family_id in test_set else "train")
            for family in catalog.families
        }
        split = cls(
            seed=seed,
            assignment=assignment,
            fractions=None,
            catalog_fingerprint=catalog_fingerprint(catalog),
            policy=POLICY_COUNTS,
            n_holdout=n_holdout,
            n_train=n_families - n_holdout,
        )
        assert_no_leakage(catalog, split)
        return split

# ---------------------------------------------------------------------------
# Catalog fingerprint
# ---------------------------------------------------------------------------

def catalog_fingerprint(catalog: DpfCatalog) -> str:
    """sha256 over the sorted (family_id, seqres) pairs of a catalog."""
    digest = hashlib.sha256()
    for family in sorted(catalog.families, key=lambda f: f.family_id):
        digest.update(family.family_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(family.seqres.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()

# ---------------------------------------------------------------------------
# Identity grouping
# ---------------------------------------------------------------------------

def pdb_entry_id(family_id: str) -> str | None:
    """``1abc_A`` → ``1abc``; ``None`` when the id is not ``<entry>_<chain>``.

    The shape check is what keeps rule 3 honest: ``dpf_000`` and ``dpf_011``
    share the prefix ``dpf`` but are not two chains of one deposition, and
    unioning on that prefix collapses a whole catalog into one component.
    """
    entry, sep, chain = family_id.partition("_")
    if not sep:
        return None
    if not PDB_ENTRY_ID_RE.match(entry) or not PDB_CHAIN_ID_RE.match(chain):
        return None
    return entry

def _kmers(seqres: str) -> frozenset[str]:
    seq = seqres.strip().upper()
    if len(seq) <= KMER_SIZE:
        # Too short to k-merise: the whole sequence is its own single token, so
        # only an exact match can reach the containment threshold.
        return frozenset({seq})
    return frozenset(
        seq[i : i + KMER_SIZE] for i in range(len(seq) - KMER_SIZE + 1)
    )

def _kmer_containment(left: frozenset[str], right: frozenset[str]) -> float:
    denom = min(len(left), len(right))
    if denom == 0:
        return 0.0
    return len(left & right) / denom

def _is_substring_pair(left: str, right: str) -> bool:
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    if len(short) < MIN_SUBSTRING_LENGTH:
        return False
    return short in long

def identity_edges(catalog: DpfCatalog) -> list[tuple[str, str, str]]:
    """All (family_a, family_b, reason) pairs that must share a split side.

    Deterministic, dependency-free, and O(n^2)-safe: candidate homolog pairs come
    from an inverted k-mer index, so only families that already share a k-mer are
    compared in full.
    """
    families: Sequence[DpfFamily] = list(catalog.families)
    edges: list[tuple[str, str, str]] = []

    # (1) shared on-disk structures.
    key_owner: dict[tuple[str, str], tuple[str, str]] = {}
    for family in families:
        for member in family.members:
            for key in member.containment_keys():
                seen = key_owner.get(key)
                if seen is None:
                    key_owner[key] = (family.family_id, member.member_id)
                elif seen[0] != family.family_id:
                    edges.append(
                        (
                            seen[0],
                            family.family_id,
                            f"shared structure key kind={key[0]!r} value={key[1]!r} "
                            f"({seen[0]}/{seen[1]} and "
                            f"{family.family_id}/{member.member_id})",
                        )
                    )

    # (2) identical seqres.
    seq_owner: dict[str, str] = {}
    for family in families:
        owner = seq_owner.setdefault(family.seqres, family.family_id)
        if owner != family.family_id:
            edges.append((owner, family.family_id, "identical seqres"))

    # (3) same PDB entry (different chains of one deposition).
    entry_owner: dict[str, str] = {}
    for family in sorted(families, key=lambda f: f.family_id):
        entry = pdb_entry_id(family.family_id)
        if entry is None:
            continue
        owner = entry_owner.setdefault(entry, family.family_id)
        if owner != family.family_id:
            edges.append(
                (owner, family.family_id, f"shared PDB entry id {entry!r}")
            )

    # (4) sequence homology (substring or high k-mer containment).
    edges.extend(_homology_edges(families))
    return edges

def _homology_edges(families: Sequence[DpfFamily]) -> list[tuple[str, str, str]]:
    n = len(families)
    if n < 2:
        return []
    kmer_sets = [_kmers(family.seqres) for family in families]
    index: dict[str, list[int]] = {}
    for i, kmers in enumerate(kmer_sets):
        for kmer in kmers:
            index.setdefault(kmer, []).append(i)

    if n <= KMER_INDEX_ALL_PAIRS_MAX_FAMILIES:
        # Small catalog: compare every candidate pair. Capping by a fraction of
        # ``n`` here would drop the postings of a whole homolog cluster.
        max_postings = n
    else:
        max_postings = max(MIN_KMER_POSTINGS, int(n * MAX_KMER_FAMILY_FRACTION))
    candidates: set[tuple[int, int]] = set()
    for posting in index.values():
        if len(posting) < 2 or len(posting) > max_postings:
            continue
        for a in range(len(posting)):
            for b in range(a + 1, len(posting)):
                candidates.add((posting[a], posting[b]))

    edges: list[tuple[str, str, str]] = []
    for i, j in sorted(candidates):
        left, right = families[i], families[j]
        if _is_substring_pair(left.seqres, right.seqres):
            edges.append(
                (
                    left.family_id,
                    right.family_id,
                    "one seqres is contained in the other "
                    f"(lengths {len(left.seqres)} and {len(right.seqres)})",
                )
            )
            continue
        containment = _kmer_containment(kmer_sets[i], kmer_sets[j])
        if containment >= KMER_CONTAINMENT_THRESHOLD:
            edges.append(
                (
                    left.family_id,
                    right.family_id,
                    f"{KMER_SIZE}-mer containment {containment:.3f} >= "
                    f"{KMER_CONTAINMENT_THRESHOLD}",
                )
            )
    return edges

def identity_components(catalog: DpfCatalog) -> dict[str, str]:
    """Union-find over :func:`identity_edges`: family_id → component root."""
    parent = {family.family_id: family.family_id for family in catalog.families}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_l, root_r = find(left), find(right)
        if root_l == root_r:
            return
        if root_l < root_r:
            parent[root_r] = root_l
        else:
            parent[root_l] = root_r

    for left, right, _reason in identity_edges(catalog):
        union(left, right)
    return {family.family_id: find(family.family_id) for family in catalog.families}

def sequence_components(catalog: DpfCatalog) -> dict[str, str]:
    """Backwards-compatible alias of :func:`identity_components`.

    Kept because callers used to think of the split atom as "the seqres"; it is
    now the whole biological identity component (structure, entry id, homology).
    """
    return identity_components(catalog)

def _groups_by_root(components: dict[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for family_id, root in sorted(components.items()):
        groups.setdefault(root, []).append(family_id)
    return groups

# ---------------------------------------------------------------------------
# Quota assignment
# ---------------------------------------------------------------------------

def _shuffled_roots(seed: int, roots: Iterable[str]) -> list[str]:
    """Deterministic seeded shuffle of the component roots."""
    return sorted(roots, key=lambda root: (_unit_interval(seed, root), root))

def quota_assignment(
    groups: dict[str, list[str]],
    fractions: SplitFractions,
    seed: int,
) -> dict[str, str]:
    """Assign whole components to splits so realised sizes match ``fractions``.

    Pure function of ``(seed, groups, fractions)``. Components are shuffled with
    the seed, then filled by largest remaining family deficit, so the realised
    family counts are as close to the requested fractions as whole components
    allow. Every split with a strictly positive fraction gets at least one whole
    component -- the smallest one available, so the reservation cannot overshoot
    the quota -- and it is a hard error (not a warning) to ask for more non-empty
    splits than there are components.

    NOTE: this is quota-based, so it is *not* stable under catalog changes --
    adding or removing a family can move other families between splits. That is
    why the persisted payload carries a catalog fingerprint (see
    ``DpfSplit.load``), which turns a stale split into a loud error.
    """
    order = _shuffled_roots(seed, groups)
    weights = {root: len(members) for root, members in groups.items()}
    total = float(sum(weights.values()))
    names = fractions.positive_names()
    if not names:
        raise ValueError("At least one split fraction must be positive")

    targets = {name: getattr(fractions, name) * total for name in names}
    assigned = {name: 0.0 for name in names}
    out: dict[str, str] = {}

    queue = list(order)
    if len(queue) < len(names):
        raise ValueError(
            f"Cannot honour the requested split: {len(queue)} identity "
            f"component(s) available for {len(names)} non-empty split(s) "
            f"({', '.join(sorted(names))}). Every split with a strictly "
            f"positive fraction needs at least one whole component; widen the "
            f"catalog or zero out the fractions you cannot fill."
        )
    # Reserve one component per non-empty split (largest fraction first) so a
    # small catalog can never produce an empty val/test. Reserve the *smallest*
    # remaining component: popping the front of the shuffled queue can hand a
    # huge homolog cluster to val/test and blow past the quota without bound.
    # Ties keep shuffled order, so equal-size components behave as before.
    for name in sorted(names, key=lambda n: (-getattr(fractions, n), n)):
        idx = min(range(len(queue)), key=lambda i: (weights[queue[i]], i))
        root = queue.pop(idx)
        out[root] = name
        assigned[name] += weights[root]

    for root in queue:
        name = max(
            names,
            key=lambda n: (targets[n] - assigned[n], -SPLIT_NAMES.index(n)),
        )
        out[root] = name
        assigned[name] += weights[root]
    return out

def _unit_interval(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)

# ---------------------------------------------------------------------------
# Leak checks
# ---------------------------------------------------------------------------

def assert_no_leakage(catalog: DpfCatalog, split: DpfSplit) -> None:
    """Fail if any identity component crosses splits.

    Enforces exactly the grouping :func:`identity_components` builds, so a
    hand-written split cannot smuggle a homolog, a shared structure file, or a
    second chain of the same PDB entry across the split boundary.
    """
    catalog_ids = set(catalog.family_ids())
    split_ids = set(split.assignment)
    if catalog_ids != split_ids:
        raise StructuralLeakError(
            "Split and catalog family_id sets differ: "
            f"only_in_catalog={sorted(catalog_ids - split_ids)} "
            f"only_in_split={sorted(split_ids - catalog_ids)}"
        )

    unknown = {
        fid: name
        for fid, name in split.assignment.items()
        if name not in SPLIT_NAMES
    }
    if unknown:
        raise StructuralLeakError(f"Invalid split labels: {unknown}")

    for left, right, reason in identity_edges(catalog):
        left_name = split.assignment[left]
        right_name = split.assignment[right]
        if left_name != right_name:
            raise StructuralLeakError(
                f"Families {left!r} ({left_name}) and {right!r} ({right_name}) "
                f"must stay on one side of the split: {reason}."
            )
