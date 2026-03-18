"""Arranca el sistema completo sobre las 24 criptos estudiadas en modo paper."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from scripts.bootstrap_studied_universe_paper_root import bootstrap_studied_universe_paper_root


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI del runner live paper."""
    parser = argparse.ArgumentParser(
        description="Arranca main.py con las 24 criptos estudiadas en modo paper + dry-run."
    )
    parser.add_argument("--registry-root", default=settings.STUDIED_UNIVERSE_PAPER_ROOT)
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="No reconstruye el root paper antes de arrancar main.py.",
    )
    return parser


def main() -> None:
    """Arranca main.py con el universo estudiado y paper trading activado."""
    args = build_parser().parse_args()
    protected_roots = {
        Path("models/multi_asset_live_v2").as_posix(),
        Path("tests/legacy/models/multi_asset_phase2_experimental").as_posix(),
    }
    if Path(args.registry_root).as_posix() in protected_roots:
        raise ValueError("El runner live paper exige un root separado del live principal y de phase2.")
    if not args.skip_bootstrap:
        bootstrap_studied_universe_paper_root(target_root=args.registry_root)

    studied_bases_csv = ",".join(settings.STUDIED_UNIVERSE_BASES)
    env = os.environ.copy()
    env["MODEL_REGISTRY_ROOT"] = args.registry_root
    env["OBSERVED_BASES_CSV"] = studied_bases_csv
    env["EXECUTION_ALLOWED_BASES_CSV"] = studied_bases_csv
    env["PAPER_TRADING_ENABLED"] = "true"
    env["EXECUTION_DRY_RUN"] = "true"

    command = [sys.executable, str(ROOT / "main.py")]
    subprocess.run(command, check=True, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    main()
