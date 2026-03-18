"""Runner dedicado para evaluar juntas las 24 criptos estudiadas."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from scripts.bootstrap_studied_universe_paper_root import bootstrap_studied_universe_paper_root


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI del runner unificado."""
    parser = argparse.ArgumentParser(description="Corre una sesion paper con las 24 criptos estudiadas.")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--session-root", default="data/studied_universe_sessions")
    parser.add_argument("--registry-root", default=settings.STUDIED_UNIVERSE_PAPER_ROOT)
    parser.add_argument("--order-notional-usd", type=float, default=settings.PILOT_ORDER_NOTIONAL_USD)
    parser.add_argument("--paper-initial-cash", type=float, default=settings.PAPER_INITIAL_CASH)
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="No reconstruye el root paper antes de correr la sesion.",
    )
    return parser


def main() -> None:
    """Delegacion simple al runner generico con defaults del universo estudiado."""
    args = build_parser().parse_args()
    if not args.skip_bootstrap:
        bootstrap_studied_universe_paper_root(target_root=args.registry_root)

    studied_bases_csv = ",".join(settings.STUDIED_UNIVERSE_BASES)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_paper_session.py"),
        "--cohort-name",
        "studied_universe_24",
        "--seconds",
        str(args.seconds),
        "--session-root",
        args.session_root,
        "--registry-root",
        args.registry_root,
        "--observed-bases-csv",
        studied_bases_csv,
        "--allowed-bases-csv",
        studied_bases_csv,
        "--order-notional-usd",
        str(args.order_notional_usd),
        "--paper-initial-cash",
        str(args.paper_initial_cash),
        "--paper-fee-pct",
        str(settings.PAPER_FEE_PCT),
        "--paper-slippage-pct",
        "0.1",
        "--paper-slippage-overrides-csv",
        "PENGU=2.0",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
