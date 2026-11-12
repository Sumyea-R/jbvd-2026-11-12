from pathlib import Path


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / ".git").exists() or (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src" / "sustainability").is_dir()
        ):
            return candidate
    return start.resolve()


def get_questions_path(repo_root: Path, filename: str) -> Path:
    new_path = repo_root / "src" / "sustainability" / "config" / filename
    if not new_path.exists():
        raise FileNotFoundError(f"Missing config file: {new_path}")
    return new_path


def get_outputs_root(repo_root: Path) -> Path:
    path = repo_root / "private" / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_root(repo_root: Path) -> Path:
    path = repo_root / "private" / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path
