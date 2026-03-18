"""Descarga y entrena un modelo separado por cripto."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings

LEGACY_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "UNI/USDT",
]
PHASE_ONE_SYMBOLS = list(settings.PHASE_ONE_TRAINING_SYMBOLS)
PHASE_TWO_SYMBOLS = list(settings.PHASE_TWO_EXPERIMENTAL_TRAINING_SYMBOLS)
ALL_SYMBOLS = LEGACY_SYMBOLS + PHASE_ONE_SYMBOLS
ALL_EXPERIMENTAL_SYMBOLS = PHASE_ONE_SYMBOLS + PHASE_TWO_SYMBOLS


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI."""
    parser = argparse.ArgumentParser(description="Entrena un modelo separado para multiples criptos.")
    parser.add_argument("--exchange", default="binanceus", help="Exchange CCXT para descarga historica.")
    parser.add_argument(
        "--preset",
        choices=("legacy", "phase1", "phase2", "all", "all_experimental"),
        default="legacy",
        help="Grupo de activos predefinido a usar si no se pasan --symbols.",
    )
    parser.add_argument("--symbols", nargs="+", default=None, help="Lista de simbolos.")
    parser.add_argument("--timeframe", default="1m", help="Timeframe OHLCV.")
    parser.add_argument("--candles", type=int, default=259200, help="Cantidad de velas por simbolo.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Tamano de lote para fetch_ohlcv.")
    parser.add_argument("--output-dir", default="data/historical_multi", help="Directorio de CSVs.")
    parser.add_argument("--registry-root", default="models/multi_asset", help="Raiz de directorios de modelos.")
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Ruta opcional donde guardar el resumen JSON consolidado.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Cantidad de folds walk-forward.")
    parser.add_argument("--min-train-size", type=int, default=1000, help="Minimo de train por fold.")
    return parser


def main() -> None:
    """Descarga y entrena secuencialmente cada simbolo."""
    args = build_parser().parse_args()
    summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    symbols = args.symbols or _resolve_symbols_from_preset(args.preset)

    for symbol in symbols:
        safe_symbol = symbol.replace("/", "_").lower()
        product_id = symbol.replace("/", "-")
        csv_path = Path(args.output_dir) / f"{safe_symbol}_{args.timeframe}_{args.exchange}.csv"
        registry_dir = Path(args.registry_root) / safe_symbol

        try:
            run_subprocess(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "fetch_coinbase_history.py"),
                    "--exchange",
                    args.exchange,
                    "--symbols",
                    symbol,
                    "--timeframe",
                    args.timeframe,
                    "--candles",
                    str(args.candles),
                    "--batch-size",
                    str(args.batch_size),
                    "--output-dir",
                    args.output_dir,
                ]
            )

            retrain_stdout = run_subprocess(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "retrain_from_historical.py"),
                    "--path",
                    str(csv_path),
                    "--product-id",
                    product_id,
                    "--registry-dir",
                    str(registry_dir),
                    "--n-splits",
                    str(args.n_splits),
                    "--min-train-size",
                    str(args.min_train_size),
                ],
                capture_output=True,
            )
            summary = extract_json_summary(retrain_stdout)
            summary["symbol"] = symbol
            summary["csv_path"] = str(csv_path)
            summaries.append(summary)
            print(json.dumps(summary, indent=2))
        except subprocess.CalledProcessError as exc:
            failure = {
                "symbol": symbol,
                "stage": "fetch_or_train",
                "error": str(exc),
            }
            failures.append(failure)
            print(json.dumps(failure, indent=2))

    failure_ratio = (len(failures) / len(symbols)) if symbols else 0.0
    final_summary = {
        "completed": len(summaries),
        "failed": len(failures),
        "failure_ratio": failure_ratio,
        "symbols": summaries,
        "failures": failures,
        "warning": "partial_failures_detected" if failures else "",
    }
    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(json.dumps(final_summary, indent=2))
    if failure_ratio > 0.30:
        raise SystemExit(1)


def _resolve_symbols_from_preset(preset: str) -> list[str]:
    """Resuelve la lista de simbolos a partir del preset solicitado."""
    preset_map = {
        "legacy": LEGACY_SYMBOLS,
        "phase1": PHASE_ONE_SYMBOLS,
        "phase2": PHASE_TWO_SYMBOLS,
        "all": ALL_SYMBOLS,
        "all_experimental": ALL_EXPERIMENTAL_SYMBOLS,
    }
    return list(preset_map[preset])


def run_subprocess(command: list[str], capture_output: bool = False) -> str:
    """Ejecuta un subproceso y propaga errores."""
    result = subprocess.run(command, check=True, text=True, capture_output=capture_output)
    if capture_output:
        return result.stdout
    return ""


def extract_json_summary(stdout: str) -> dict[str, object]:
    """Extrae el ultimo objeto JSON de la salida del reentrenamiento."""
    start = stdout.rfind("{")
    if start < 0:
        raise ValueError("No se encontro resumen JSON en la salida.")
    return json.loads(stdout[start:])


if __name__ == "__main__":
    main()
