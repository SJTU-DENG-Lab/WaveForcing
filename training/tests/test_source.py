"""CPU provenance checks without Git mutations, model imports, or GPU use."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from wf_training.cli import _run, main
from wf_training.config import resolve_config
from wf_training.utils.source import source_provenance


class SourceProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "wf_training"
        (self.package / "configs").mkdir(parents=True)
        (self.package / "__init__.py").write_text("# package\n")
        (self.package / "new_untracked.py").write_text("value = 14\n")
        (self.package / "configs/14b.yaml").write_text("world_size: 8\n")

    def test_records_git_identity_and_every_current_source_hash(self):
        with patch("wf_training.utils.source._git_value", side_effect=[
            str(self.root), "a" * 40, "train/14b-fsdp8",
        ]):
            report = source_provenance(self.package)
        self.assertEqual(report["git_commit"], "a" * 40)
        self.assertEqual(report["git_branch"], "train/14b-fsdp8")
        self.assertEqual(report["repository_root"], str(self.root))
        self.assertEqual(set(report["files_sha256"]), {
            "__init__.py", "new_untracked.py", "configs/14b.yaml",
        })
        self.assertEqual(report["files_sha256"]["new_untracked.py"],
                         hashlib.sha256(b"value = 14\n").hexdigest())

    def test_fingerprint_changes_for_source_edits_but_not_other_assets(self):
        with patch("wf_training.utils.source._git_value", return_value=None):
            first = source_provenance(self.package)
            (self.package / "model.pt").write_text("weights are not source")
            (self.package / "data.json").write_text("{}")
            self.assertEqual(first, source_provenance(self.package))
            (self.package / "new_untracked.py").write_text("value = 15\n")
            changed = source_provenance(self.package)
        self.assertNotEqual(first["source_sha256"], changed["source_sha256"])

    def test_missing_git_or_not_a_repository_still_hashes_source(self):
        for failure in (FileNotFoundError("git is absent"),
                        subprocess.TimeoutExpired("git", 5)):
            with self.subTest(failure=failure), patch(
                    "wf_training.utils.source.subprocess.run", side_effect=failure):
                report = source_provenance(self.package)
            self.assertIsNone(report["git_commit"])
            self.assertIsNone(report["git_branch"])
            self.assertIsNone(report["repository_root"])
            self.assertEqual(len(report["files_sha256"]), 3)
        with patch("wf_training.utils.source.subprocess.run", return_value=
                   subprocess.CompletedProcess(["git"], 128, stdout="", stderr="not a repository")):
            self.assertIsNone(source_provenance(self.package)["git_commit"])

    def test_hidden_paths_and_links_are_not_read(self):
        hidden = self.package / ".private"
        hidden.mkdir()
        (hidden / "secret.py").write_text("synthetic private fixture")
        outside = self.root / "outside.py"
        outside.write_text("synthetic external fixture")
        (self.package / "linked.py").symlink_to(outside)
        (self.package / "linked_directory").symlink_to(hidden, target_is_directory=True)
        with patch("wf_training.utils.source._git_value", return_value=None):
            report = source_provenance(self.package)
        self.assertEqual(len(report["files_sha256"]), 3)
        self.assertEqual(report["skipped_symlinks"], ["linked.py"])

    def test_run_manifest_includes_source_provenance(self):
        config = resolve_config("s1", {
            "model_root": str(self.root / "models"), "prompts": str(self.root / "prompts.txt"),
            "distill_init_14b": str(self.root / "candidate.pt"),
        }, self.root / "run", recipe="14b-fsdp8-smoke")
        provenance = {"git_commit": "b" * 40, "files_sha256": {"new.py": "hash"}}
        report = {"assets": {"initial_checkpoint": {"path": config.generator_ckpt}}}
        with patch("wf_training.cli.preflight", return_value=report), patch(
                "wf_training.cli.source_provenance", return_value=provenance), patch(
                "wf_training.cli._execute"):
            _run([config])
        manifest = json.loads((Path(config.logdir) / "run_manifest.json").read_text())
        self.assertEqual(manifest["source_provenance"], provenance)

    def test_dry_run_does_not_probe_git_or_hash_source(self):
        assets = self.root / "assets.yaml"
        OmegaConf.save(OmegaConf.create({
            "model_root": "/missing/models", "prompts": "/missing/prompts.txt",
            "distill_init_14b": "/missing/candidate.pt",
        }), assets)
        with patch("wf_training.cli.source_provenance", side_effect=AssertionError("source scan")), \
                redirect_stdout(io.StringIO()):
            result = main([
                "run", "--recipe", "14b-fsdp8-smoke", "--stage", "s1",
                "--assets", str(assets), "--output", str(self.root / "dry"), "--dry-run",
            ])
        self.assertEqual(result, 0)
        self.assertFalse((self.root / "dry").exists())


if __name__ == "__main__":
    unittest.main()
