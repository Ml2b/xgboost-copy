"""Ejecuta un screener de operabilidad sobre el universo spot de Coinbase USA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exchange.coinbase_client import CoinbaseAdvancedTradeClient
from exchange.operability_screener import (
    CoinbaseOperabilityScreener,
    OperabilityConfig,
    OperabilityScreenReport,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screening de activos operables en Coinbase USA.")
    parser.add_argument("--min-volume-usd", type=float, default=1_000_000.0)
    parser.add_argument("--max-spread-pct", type=float, default=0.0025)
    parser.add_argument("--min-age-days", type=int, default=45)
    parser.add_argument("--lookback-minutes", type=int, default=300)
    parser.add_argument("--min-candle-coverage-pct", type=float, default=0.90)
    parser.add_argument("--max-candle-checks", type=int, default=80)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--output-json", type=Path, default=Path("reports/coinbase_operability_screen.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/coinbase_operability_screen.md"))
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def render_markdown(report: OperabilityScreenReport, top: int) -> str:
    train_now = [result for result in report.results if result.tier == "train_now"][:top]
    observe = [result for result in report.results if result.tier == "observe"][:top]

    lines = [
        "# Screening Coinbase USA",
        "",
        f"- Generado: `{report.generated_at}`",
        f"- Productos spot vistos: `{report.total_products_seen}`",
        f"- Activos base unicos vistos: `{report.total_base_assets_seen}`",
        f"- Candidatos evaluados: `{report.candidate_base_assets}`",
        f"- `train_now`: `{sum(1 for result in report.results if result.tier == 'train_now')}`",
        f"- `observe`: `{sum(1 for result in report.results if result.tier == 'observe')}`",
        f"- `skip`: `{sum(1 for result in report.results if result.tier == 'skip')}`",
        "",
        "## Train Now",
        "",
        "| Base | Producto | Score | Vol 24h USD | Spread % | Edad dias | Cobertura 1m | Razones |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in train_now:
        lines.append(
            "| {base} | {product} | {score:.2f} | {volume:,.0f} | {spread:.4f} | {age} | {coverage:.2%} | {reasons} |".format(
                base=result.base_asset,
                product=result.product_id,
                score=result.score,
                volume=result.quote_volume_24h,
                spread=0.0 if result.spread_pct is None else (result.spread_pct * 100.0),
                age=result.listing_age_days,
                coverage=result.candle_coverage_pct,
                reasons=", ".join(result.reasons),
            )
        )

    lines.extend(
        [
            "",
            "## Observe",
            "",
            "| Base | Producto | Score | Vol 24h USD | Spread % | Edad dias | Cobertura 1m | Razones |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in observe:
        lines.append(
            "| {base} | {product} | {score:.2f} | {volume:,.0f} | {spread:.4f} | {age} | {coverage:.2%} | {reasons} |".format(
                base=result.base_asset,
                product=result.product_id,
                score=result.score,
                volume=result.quote_volume_24h,
                spread=0.0 if result.spread_pct is None else (result.spread_pct * 100.0),
                age=result.listing_age_days,
                coverage=result.candle_coverage_pct,
                reasons=", ".join(result.reasons),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    client = CoinbaseAdvancedTradeClient()
    config = OperabilityConfig(
        min_quote_volume_24h=args.min_volume_usd,
        max_spread_pct=args.max_spread_pct,
        min_listing_age_days=args.min_age_days,
        candle_lookback_minutes=args.lookback_minutes,
        min_candle_coverage_pct=args.min_candle_coverage_pct,
        max_candidates_for_candles=args.max_candle_checks,
    )
    report = CoinbaseOperabilityScreener(client=client, config=config).screen_assets()

    ensure_parent(args.output_json)
    ensure_parent(args.output_md)
    args.output_json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(report, top=args.top), encoding="utf-8")

    top_results = [result.to_dict() for result in report.results[: args.top]]
    summary = report.to_dict()["summary"]
    print(
        json.dumps(
            {
                "generated_at": report.generated_at,
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "summary": summary,
                "top_results": top_results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
