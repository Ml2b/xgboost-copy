"""Bootstrap seguro para Render con modelos seed y almacenamiento persistente."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED_MODEL_ROOT = ROOT / "seed_models" / "multi_asset_paper_24"
LOCK_PATH = ROOT / "runtime" / ".seed_copy.lock"


def main() -> None:
    """Prepara el entorno persistente y arranca el worker principal."""
    model_root = Path(
        os.getenv(
            "MODEL_REGISTRY_ROOT",
            str(ROOT / "runtime" / "models" / "multi_asset_paper_24"),
        )
    )
    history_db_path = Path(
        os.getenv(
            "TRAINER_HISTORY_DB_PATH",
            str(ROOT / "runtime" / "trainer_candle_history.sqlite3"),
        )
    )

    history_db_path.parent.mkdir(parents=True, exist_ok=True)
    model_root.parent.mkdir(parents=True, exist_ok=True)
    ensure_seed_models(model_root)

    command = [sys.executable, str(ROOT / "main.py")]
    subprocess.run(command, check=True, cwd=str(ROOT), env=os.environ.copy())


def ensure_seed_models(target_root: Path) -> None:
    """Copia el seed de modelos una sola vez al storage persistente de Render."""
    if target_root.exists() and any(target_root.iterdir()):
        return
    if not SEED_MODEL_ROOT.exists():
        raise FileNotFoundError(f"No existe el seed de modelos: {SEED_MODEL_ROOT}")

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = acquire_lock()
    try:
        if target_root.exists() and any(target_root.iterdir()):
            return

        temp_target = target_root.parent / f"{target_root.name}.tmp"
        if temp_target.exists():
            shutil.rmtree(temp_target)
        shutil.copytree(SEED_MODEL_ROOT, temp_target)
        if target_root.exists():
            shutil.rmtree(target_root)
        temp_target.rename(target_root)
    finally:
        release_lock(lock_fd)


def acquire_lock(timeout_seconds: int = 30) -> int:
    """Toma un lockfile simple para evitar doble copia del seed."""
    started = time.monotonic()
    while True:
        try:
            return os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            if time.monotonic() - started >= timeout_seconds:
                raise TimeoutError("No se pudo adquirir el lock de seed models en Render.")
            time.sleep(0.5)


def release_lock(lock_fd: int) -> None:
    """Libera el lockfile del bootstrap."""
    os.close(lock_fd)
    with contextlib.suppress(FileNotFoundError):
        LOCK_PATH.unlink()


if __name__ == "__main__":
    main()
