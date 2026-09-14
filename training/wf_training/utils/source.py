"""Read-only provenance for the installed training package, using the stdlib."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


def _git_value(directory: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(directory), *arguments],
            capture_output=True, text=True, check=False, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    return result.stdout.strip() or None


def source_provenance(package_root: str | Path | None = None) -> dict:
    """Hash all package Python/YAML files, including untracked source.

    Git identifies the base revision; the file hashes identify the actual
    mutable source. Hidden paths and symlinks are excluded to avoid following
    source-tree links into unrelated private files. No diff or environment is
    captured, and no files or Git state are changed.
    """
    root = (Path(package_root) if package_root is not None
            else Path(__file__).resolve().parents[1]).resolve()
    files = {}
    skipped_symlinks = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.suffix not in (".py", ".yaml"):
            continue
        if path.is_symlink():
            skipped_symlinks.append(relative.as_posix())
            continue
        if path.is_file():
            files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    aggregate = hashlib.sha256()
    for name, digest in files.items():
        aggregate.update(f"{name}\0{digest}\n".encode())
    return {
        "package_root": str(root),
        "repository_root": _git_value(root, "rev-parse", "--show-toplevel"),
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_branch": _git_value(root, "symbolic-ref", "--quiet", "--short", "HEAD"),
        "source_sha256": aggregate.hexdigest(),
        "files_sha256": files,
        "skipped_symlinks": skipped_symlinks,
    }
