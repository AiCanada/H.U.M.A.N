# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shipping a DPF run to a rented Linux GPU without changing what it trains on.

Two invariants carry the whole migration.

**The fingerprint.** ``DpfSplit.load`` hashes the sorted ``(family_id, seqres)`` pairs and
refuses a split built from a different catalog. That is what makes a cross-OS resume safe --
the restart state in a checkpoint is path-free, so the bag is *rebuilt* from the catalog,
and the fingerprint is the guarantee that it rebuilds identically. Every trimming decision
here has to leave it untouched.

**The trap.** Dropping the 5 test families to save 2.5 GiB of trajectories changes the
catalog to 81 families and the fingerprint with it. Keeping their topology PDBs -- 1.4 MB --
keeps the corpus at 86 and the fingerprint byte-identical, while their trajectories stay
home because ``fit`` never loads the test split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rbase.data.dpf.catalog import DpfCatalog
from rbase.data.dpf.split import catalog_fingerprint

from .toys import make_atlas_family, make_family

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

stage_remote_payload = pytest.importorskip("stage_remote_payload")
sync_checkpoints_hf = pytest.importorskip("sync_checkpoints_hf")

def build_store(tmp_path: Path, ids: tuple[str, ...]) -> tuple[Path, DpfCatalog]:
    """An ATLAS-shaped store with real (empty) trajectory files on disk."""
    root = tmp_path / "dpf"
    families = []
    for i, family_id in enumerate(ids):
        family = make_atlas_family(root, family_id, "AGSVLE" * (2 + i))
        for member in family.members:
            if member.xtc_path is not None:
                Path(member.xtc_path).write_bytes(b"\x00" * 64)
        families.append(family)
    return root, DpfCatalog(families=families)

# --------------------------------------------------------------------------
# The shell trap
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mangled",
    [
        "C:/Program Files/Git/workspace/rbase_data",
        r"C:\workspace\rbase_data",
        "workspace/rbase_data",
    ],
)
def test_remote_root_rejects_a_shell_mangled_path(mangled):
    """Git Bash rewrites '/workspace/x' before python sees it.

    The rewritten value is baked into every member path in catalog.json, resolves
    to nothing on the instance, and surfaces only after a 43 GiB upload. Caught at
    the point where it costs nothing to fix.
    """
    with pytest.raises(SystemExit) as excinfo:
        stage_remote_payload.validate_remote_root(mangled)
    assert "POSIX absolute path" in str(excinfo.value)

def test_remote_root_accepts_the_msys_escape():
    """'//workspace/x' is how MSYS suppresses its own rewriting."""
    assert stage_remote_payload.validate_remote_root("//workspace/x") == "/workspace/x"
    assert stage_remote_payload.validate_remote_root("/workspace/x/") == "/workspace/x"

# --------------------------------------------------------------------------
# Rebasing paths without disturbing the corpus
# --------------------------------------------------------------------------

def test_catalog_paths_are_rebased_onto_the_instance(tmp_path):
    root, catalog = build_store(tmp_path, ("1abc_A", "2def_B"))
    payload = stage_remote_payload.remote_catalog_dict(
        catalog, root.resolve(), "/workspace/rbase_data"
    )
    paths = [
        value
        for family in payload["families"]
        for member in family["members"]
        for key, value in member.items()
        if key.endswith("_path") or key == "xtc_top_pdb"
    ]
    assert paths, "catalog declared no member paths"
    for path in paths:
        assert path.startswith("/workspace/rbase_data/dpf/")
        assert "\\" not in path, "Windows separators would not resolve on Linux"

def test_rebasing_does_not_change_the_fingerprint(tmp_path):
    """Paths move; (family_id, seqres) must not.

    The fingerprint is what the resume is checked against, so a rebase that
    perturbed it would silently invalidate the checkpoint.
    """
    root, catalog = build_store(tmp_path, ("1abc_A", "2def_B", "3ghi_C"))
    before = catalog_fingerprint(catalog)

    payload = stage_remote_payload.remote_catalog_dict(
        catalog, root.resolve(), "/workspace/rbase_data"
    )
    rebuilt = DpfCatalog(
        families=[
            type(catalog.families[0])(
                family_id=family["family_id"],
                seqres=family["seqres"],
                members=catalog.by_id()[family["family_id"]].members,
            )
            for family in payload["families"]
        ]
    )
    assert catalog_fingerprint(rebuilt) == before

# --------------------------------------------------------------------------
# What actually ships
# --------------------------------------------------------------------------

def test_test_families_keep_topology_but_not_trajectories(tmp_path):
    """The 2.5 GiB saving, and the 1.4 MB that makes it safe."""
    root, catalog = build_store(tmp_path, ("1abc_A", "2def_B", "3ghi_C"))
    assignment = {"1abc_A": "train", "2def_B": "val", "3ghi_C": "test"}
    out = tmp_path / "staged"

    staged, tally = stage_remote_payload.stage_families(catalog, assignment, root, out)

    assert (out / "3ghi_C" / "protein" / "3ghi_C.pdb").is_file(), (
        "the test family's topology is what keeps the catalog at 3 families"
    )
    assert not list((out / "3ghi_C" / "protein").glob("*.xtc")), (
        "test trajectories are never loaded during fit and must stay home"
    )
    for family_id in ("1abc_A", "2def_B"):
        assert list((out / family_id / "protein").glob("*.xtc")), (
            f"{family_id} is trainable and needs its trajectories"
        )
    assert tally["xtc"] > 0 and tally["pdb"] > 0
    assert len(staged) == len([p for p in staged if p.is_file()])

def test_dropping_test_directories_would_break_the_resume(tmp_path):
    """Why the previous test keeps those PDBs at all.

    Shipping only train+val *directories* is the obvious reading of "only transfer
    train and val proteins", and it silently changes the corpus the split was built
    against. The failure is loud at load time, but the cost is a repeated upload.
    """
    _, catalog = build_store(tmp_path, ("1abc_A", "2def_B", "3ghi_C"))
    full = catalog_fingerprint(catalog)
    without_test = catalog_fingerprint(catalog.select(["1abc_A", "2def_B"]))
    assert without_test != full

def test_staging_is_readonly_on_the_source(tmp_path):
    """Hardlinking must not give the staging tree write-through to the source."""
    root, catalog = build_store(tmp_path, ("1abc_A",))
    src = root / "1abc_A" / "protein" / "1abc_A.pdb"
    before = src.read_bytes()
    stage_remote_payload.stage_families(
        catalog, {"1abc_A": "train"}, root, tmp_path / "staged"
    )
    assert src.read_bytes() == before

# --------------------------------------------------------------------------
# Pushing checkpoints to the Hub and freeing billed disk
# --------------------------------------------------------------------------

def test_step_is_read_from_the_filename():
    assert sync_checkpoints_hf.step_of("dpf-epoch001-step00001500.ckpt") == 1500
    assert sync_checkpoints_hf.step_of("dpf-bestfwd-step00000200.ckpt") == 200
    # No step: sorts first, and is protected from deletion anyway.
    assert sync_checkpoints_hf.step_of("last.ckpt") == -1

def test_last_ckpt_is_never_deleted():
    """The running job rewrites it every save and --resume last/auto target it."""
    assert "last.ckpt" in sync_checkpoints_hf.PROTECTED
    assert "restart.json" in sync_checkpoints_hf.PROTECTED

def test_only_checkpoint_artifacts_are_touched():
    """Freeing disk must not reach the logs, the manifest, or the training data."""
    keep = [
        "dpf-epoch001-step00001500.ckpt",
        "dpf-epoch001-step00001500.restart.json",
        "restart.json",
        "last.ckpt",
    ]
    ignore = ["events.out.tfevents.123", "run_manifest.json", "STOP", "config.yaml"]
    for name in keep:
        assert sync_checkpoints_hf.is_checkpoint_artifact(name), name
    for name in ignore:
        assert not sync_checkpoints_hf.is_checkpoint_artifact(name), name

class _StubHub:
    """Minimal HfApi stand-in: records uploads, reports back only what it stored."""

    def __init__(self, reject: set[str] | None = None):
        self.stored: dict[str, int] = {}
        self.reject = reject or set()

    def upload_file(self, *, path_or_fileobj, path_in_repo, **_):
        name = Path(path_in_repo).name
        if name in self.reject:
            raise RuntimeError("simulated upload failure")
        self.stored[name] = Path(path_or_fileobj).stat().st_size

    def list_repo_tree(self, **_):
        return [
            type("Entry", (), {"path": f"checkpoints/{n}", "size": s})()
            for n, s in self.stored.items()
        ]

def test_prune_is_refused_on_windows(tmp_path, monkeypatch, capsys):
    """Local run output stays in the local folder; cloud output goes to the Hub.

    Pruning only earns its keep on the rented box, where a deleted local copy frees
    storage billed per GB-month. Pointed at the Windows working copy the same flag
    would upload the local run's history and then delete it.
    """
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    survivor = ckpt / "dpf-epoch000-step00000150.ckpt"
    survivor.write_bytes(b"\x00" * 32)

    monkeypatch.setattr(sync_checkpoints_hf.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        sys, "argv", ["sync_checkpoints_hf.py", "--ckpt_dir", str(ckpt), "--prune"]
    )

    assert sync_checkpoints_hf.main() == 1
    assert survivor.is_file(), "the local checkpoint must survive the refusal"
    assert "Refusing to --prune on Windows" in capsys.readouterr().err

def test_a_checkpoint_the_hub_never_confirmed_is_not_deleted(tmp_path, monkeypatch):
    """The whole point of reading sizes back instead of trusting the upload call.

    An upload that raises must leave the only copy of that checkpoint on disk;
    deleting it would lose the work the run is being checkpointed to protect.
    """
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for name in (
        "dpf-epoch000-step00000500.ckpt",
        "dpf-epoch000-step00001000.ckpt",
        "dpf-epoch000-step00001500.ckpt",
        "last.ckpt",
    ):
        (ckpt_dir / name).write_bytes(b"\x00" * 2048)

    doomed = "dpf-epoch000-step00000500.ckpt"
    hub = _StubHub(reject={doomed})
    monkeypatch.setattr(sync_checkpoints_hf, "HfHubHTTPError", RuntimeError)

    sync_checkpoints_hf.push_once(
        hub, ckpt_dir, "AICanada/H.U.M.A.N", "dataset", "checkpoints",
        keep_recent=1, prune=True,
    )

    assert (ckpt_dir / doomed).is_file(), "an unconfirmed checkpoint must survive"
    assert (ckpt_dir / "last.ckpt").is_file(), "last.ckpt is protected"
    assert (ckpt_dir / "dpf-epoch000-step00001500.ckpt").is_file(), (
        "the newest step is kept so a crash resumes without a download"
    )
    assert not (ckpt_dir / "dpf-epoch000-step00001000.ckpt").exists(), (
        "a confirmed, non-recent checkpoint is what we free disk with"
    )

# --------------------------------------------------------------------------
# The bootstrap's own train command must actually parse
# --------------------------------------------------------------------------

CONTINUATION = "\\" + "\n"

#: Shell variables the bootstrap passes where an integer is required.
_INT_SHELL_VARS = frozenset({"WORKERS"})

def _train_invocations(script: str) -> list[list[str]]:
    """Every `rbase train ...` call in the bootstrap, as argv lists.

    The script is checked out with CRLF on Windows, so continuations have to be
    normalised before they can be joined -- otherwise the trailing backslashes
    survive into shlex and it raises on the dangling escape.
    """
    import re
    import shlex

    text = script.replace("\r\n", "\n")
    out = []
    for block in re.findall(r"rbase train(.*?)(?=\n\S|\Z)", text, re.S):
        flat = block.replace(CONTINUATION, " ")
        # Shell vars must be replaced by something of the right *type*, not a
        # generic token: --num_data_workers "$WORKERS" has to stay an int or the
        # parser rejects it for the wrong reason and the test proves nothing.
        def _fill(match: "re.Match[str]") -> str:
            name = match.group(1) or match.group(2) or ""
            return "4" if name.upper() in _INT_SHELL_VARS else "/tmp/x"

        flat = re.sub(r'"\$\{?(\w+)\}?"|\$\{?(\w+)\}?', _fill, flat)
        out.append(shlex.split(flat))
    return out

def test_the_bootstrap_train_command_parses(tmp_path):
    """Flag *names* existing is not enough -- values have formats too.

    `--forward_stride_frames 1,1024` names a real flag and passes any
    name-only check, but the parser wants `INT-INT` and rejects the comma. That
    shipped once; on a rented box it would have failed after the payload
    download, several dollars in. Run the real parser over the real command.
    """
    from rbase.cli import build_parser

    script = (REPO_ROOT / "scripts" / "vast_bootstrap.sh").read_text(encoding="utf-8")
    calls = _train_invocations(script)
    assert calls, "no `rbase train` invocation found in vast_bootstrap.sh"

    parser = build_parser()
    for argv in calls:
        # Every value is already a well-typed stand-in; just run the real parser.
        parser.parse_args(["train", *argv])

def test_forward_stride_rejects_the_comma_form():
    """Pin the exact mistake, so the format cannot regress silently."""
    import argparse

    from rbase.cli import build_parser

    parser = build_parser()
    base = ["train", "--output", "x"]
    parser.parse_args(base + ["--forward_stride_frames", "1-1024"])
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--forward_stride_frames", "1,1024"])

# --------------------------------------------------------------------------
# Catalog mode: a PDB-cluster run shipped from its catalog JSON
# --------------------------------------------------------------------------

def _static_store(tmp_path: Path):
    root = tmp_path / "pdb_clusters"
    families = [
        make_family(root, f"pdbc95_{i}", "AGSVLE" * (2 + i), member_ids=("A", "B", "C"))
        for i in range(3)
    ]
    return root, DpfCatalog(families=families)

def test_catalog_mode_stages_every_structure_by_its_own_path(tmp_path):
    """Merged families name files in several cluster directories; the payload
    mirrors each file's path under its root, which is what the rebase assumes."""
    root, catalog = _static_store(tmp_path)
    assignment = {"pdbc95_0": "train", "pdbc95_1": "val", "pdbc95_2": "test"}
    out = tmp_path / "staged"

    staged, tally = stage_remote_payload.stage_catalog_members(
        catalog, assignment, {root: "pdbc"}, out
    )

    for family in catalog.families:
        for member in family.members:
            rel = Path(member.pdb_path).resolve().relative_to(root.resolve())
            assert (out / "pdbc" / rel).is_file(), f"{family.family_id}/{member.member_id}"
    assert tally["pdb"] > 0 and tally["xtc"] == 0
    assert len(staged) == 9  # 3 families x 3 structures, test family included

    payload = stage_remote_payload.remote_catalog_dict(
        catalog, None, "/workspace/rbase_data", pdbc_root=root.resolve()
    )
    for family in payload["families"]:
        for member in family["members"]:
            assert member["pdb_path"].startswith("/workspace/rbase_data/pdbc/")
    assert catalog_fingerprint(DpfCatalog.from_dict(payload)) == catalog_fingerprint(catalog)

def test_catalog_mode_ships_trajectories_only_for_trainable_families(tmp_path):
    root, catalog = build_store(tmp_path, ("1abc_A", "2def_B"))
    assignment = {"1abc_A": "train", "2def_B": "test"}
    out = tmp_path / "staged"
    staged, tally = stage_remote_payload.stage_catalog_members(
        catalog, assignment, {root: "dpf"}, out
    )
    assert list((out / "dpf" / "1abc_A" / "protein").glob("*.xtc"))
    assert not list((out / "dpf" / "2def_B" / "protein").glob("*.xtc"))
    assert (out / "dpf" / "2def_B" / "protein" / "2def_B.pdb").is_file()
    assert tally["xtc"] > 0

def test_catalog_mode_refuses_a_member_outside_every_root(tmp_path):
    root, catalog = _static_store(tmp_path)
    with pytest.raises(ValueError, match="under none of"):
        stage_remote_payload.stage_catalog_members(
            catalog, {f.family_id: "train" for f in catalog.families},
            {tmp_path / "elsewhere": "pdbc"}, tmp_path / "staged",
        )

def test_rebasing_needs_a_root(tmp_path):
    _, catalog = _static_store(tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        stage_remote_payload.remote_catalog_dict(catalog, None, "/workspace/rbase_data")

def test_the_pdbcluster_bootstrap_pins_the_run_it_launches():
    """The instance must train the same run the local scripts define."""
    import re

    text = (REPO_ROOT / "scripts" / "vast_bootstrap_pdbcluster.sh").read_text(encoding="utf-8")
    text = re.sub(r"[ 	]+", " ", text)  # the flag table is column-aligned
    for flag in (
        "--window_frames 9",
        "--ckpt_prefix PDBcluster",
        'ONE_PASS="${ONE_PASS:-true}"',
        '--one_pass_frames "$ONE_PASS"',
        "--train_frac 0.859356 --val_frac 0.100119 --test_frac 0.040525",
        "--rescale_attention 8",
        "original_confrover_base_20m_v1_0.pt",
        "verify_remote_payload.py",
        "sync_checkpoints_hf.py",
    ):
        assert flag in text, flag
    assert "--resume epoch" not in text, "fresh run: nothing to resume from an epoch boundary"

# --- bundles: one archive instead of tens of thousands of hub requests --------

def test_bundle_round_trips_the_staged_directory(tmp_path):
    import tarfile

    out = tmp_path / "payload"
    for fam, member in (("fam_a", "1abc_A"), ("fam_a", "1abd_B"), ("fam_b", "2xyz_A")):
        p = out / "pdbc" / fam / f"{member}.pdb"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"ATOM {member}\n")
    dst, count = stage_remote_payload.write_bundle(out, "pdbc")
    assert dst == out / "bundles" / "pdbc.tar.gz" and count == 3
    with tarfile.open(dst) as tar:
        names = sorted(tar.getnames())
    # arcnames start with the directory name, so `tar -xzf ... -C $PAYLOAD` lands
    # them exactly where the per-file download would have.
    assert names == ["pdbc/fam_a/1abc_A.pdb", "pdbc/fam_a/1abd_B.pdb", "pdbc/fam_b/2xyz_A.pdb"]
    with tarfile.open(dst) as tar:
        # `filter=` only exists from python 3.11.4, and this repo still supports
        # 3.10. Feature-detected on `tarfile.data_filter`, which arrived with it,
        # rather than on a version tuple that would also have to know about the
        # backports. The archive was written four lines above, so the filter is
        # hygiene, not a guard -- worth keeping where it exists, not worth
        # failing the suite where it does not.
        safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(tmp_path / "unpacked", **safe)
    assert (tmp_path / "unpacked" / "pdbc" / "fam_b" / "2xyz_A.pdb").read_text() == "ATOM 2xyz_A\n"

def test_bundle_refuses_a_directory_that_was_not_staged(tmp_path):
    with pytest.raises(SystemExit, match="not a staged directory"):
        stage_remote_payload.write_bundle(tmp_path, "pdbc")

def test_bootstrap_prefers_bundles_and_excludes_them_from_the_snapshot():
    text = (REPO_ROOT / "scripts" / "vast_bootstrap_pdbcluster.sh").read_text(encoding="utf-8")
    assert 'HF_BUNDLES="${HF_BUNDLES:-pdbc}"' in text
    assert "bundles/$name.tar.gz" in text
    assert 'tar -xzf "$DL/$rel" -C "$PAYLOAD"' in text
    # a downloaded bundle removes its directory from the per-file snapshot
    assert 'EXCLUDES+=(--exclude "${HF_SUBDIR:+$HF_SUBDIR/}$name/*")' in text
    assert '"${EXCLUDES[@]}"' in text
    # the staging wrapper actually produces the bundle the bootstrap looks for
    ps1 = (REPO_ROOT / "runsPDB" / "stage_PDBcluster_payload.ps1").read_text(encoding="utf-8")
    assert "--bundle pdbc" in ps1

# --- the watcher must outlive the moment the checkpoint dir appears ------------

def test_wait_for_dir_polls_until_the_directory_exists(tmp_path):
    target = tmp_path / "checkpoints"
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        if len(calls) == 2:
            target.mkdir()

    sync_checkpoints_hf.wait_for_dir(target, interval=7.0, sleep=sleep)
    assert calls == [7.0, 7.0]

def test_one_shot_sync_still_refuses_a_missing_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sync_checkpoints_hf.py", "--ckpt_dir", str(tmp_path / "absent")])
    assert sync_checkpoints_hf.main() == 1
    assert "Not a directory" in capsys.readouterr().err

def test_watch_mode_waits_for_the_directory_instead_of_exiting(tmp_path, monkeypatch, capsys):
    """The bootstrap starts the watcher before training and Lightning creates
    <run>/checkpoints at the first save. On the first cloud run both watchers
    printed 'Not a directory' and exited within a second; nothing was synced."""

    class Reached(Exception):
        pass

    def fake_wait(path):
        assert path == (tmp_path / "absent").resolve()
        raise Reached

    monkeypatch.setattr(sync_checkpoints_hf, "wait_for_dir", fake_wait)
    monkeypatch.setattr(
        sys, "argv", ["sync_checkpoints_hf.py", "--ckpt_dir", str(tmp_path / "absent"), "--watch"]
    )
    with pytest.raises(Reached):
        sync_checkpoints_hf.main()
    assert "Waiting for" in capsys.readouterr().err

def test_a_prefix_nothing_was_uploaded_to_yet_reads_as_empty(monkeypatch):
    """list_repo_tree is a generator: the request, and its 404 for a folder
    that does not exist yet, happen on iteration. On the first cloud run the
    watcher failed every cycle with 'does not exist on main' and never uploaded."""

    class LazyMissing:
        def list_repo_tree(self, **_):
            def gen():
                raise RuntimeError("404 Client Error: checkpoints/x does not exist on main")
                yield  # pragma: no cover

            return gen()

    monkeypatch.setattr(sync_checkpoints_hf, "HfHubHTTPError", RuntimeError)
    assert sync_checkpoints_hf.remote_sizes(LazyMissing(), "u/r", "dataset", "checkpoints/x") == {}
