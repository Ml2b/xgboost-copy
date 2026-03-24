"""
Diagnóstico remoto del bot en Render.

Lee directamente los streams de Redis (los mismos que usa el bot en producción)
y genera un resumen de signals, dry_runs, paper fills y PnL para las últimas N horas.

Uso:
    REDIS_URL=rediss://... python scripts/diagnose_render.py
    REDIS_URL=rediss://... python scripts/diagnose_render.py --hours 6
    REDIS_URL=rediss://... python scripts/diagnose_render.py --hours 2 --tail 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import redis

from config import settings

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_redis(redis_url: str | None) -> redis.Redis:
    url = redis_url or settings.REDIS_URL
    if url:
        return redis.from_url(url, decode_responses=True, socket_connect_timeout=10)
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        username=settings.REDIS_USERNAME,
        password=settings.REDIS_PASSWORD,
        db=settings.REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=10,
    )


def _min_stream_id(hours_back: float) -> str:
    cutoff_ms = int((time.time() - hours_back * 3600) * 1000)
    return f"{cutoff_ms}-0"


def _read_stream(r: redis.Redis, stream: str, min_id: str, count: int = 10_000) -> list[dict]:
    """Lee mensajes desde min_id hasta el final del stream."""
    rows: list[dict] = []
    try:
        results = r.xrange(stream, min=min_id, count=count)
        for _msg_id, payload in results:
            rows.append(payload)
    except Exception as exc:
        print(f"  [WARN] No se pudo leer {stream}: {exc}")
    return rows


def _f(value: object, decimals: int = 2) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return str(value)


def _pct(num: int, denom: int) -> str:
    if not denom:
        return "—"
    return f"{100.0 * num / denom:.1f}%"


def _separator(char: str = "─", width: int = 70) -> str:
    return char * width


# ──────────────────────────────────────────────────────────────────────────────
# Análisis de cada stream
# ──────────────────────────────────────────────────────────────────────────────

def analyse_signals(rows: list[dict]) -> dict:
    total = len(rows)
    by_signal: Counter = Counter()
    by_asset: Counter = Counter()
    actionable = 0
    non_actionable_reasons: Counter = Counter()
    probs: list[float] = []

    for row in rows:
        sig = row.get("signal", "?").upper()
        by_signal[sig] += 1
        by_asset[row.get("product_id", "?")] += 1
        act = str(row.get("actionable", "false")).lower() == "true"
        if act:
            actionable += 1
        else:
            reason = row.get("reason", "—") or "—"
            non_actionable_reasons[reason] += 1
        try:
            probs.append(float(row["prob_buy"]))
        except Exception:
            pass

    avg_prob = sum(probs) / len(probs) if probs else 0.0
    return {
        "total": total,
        "by_signal": by_signal,
        "by_asset": by_asset,
        "actionable": actionable,
        "non_actionable_reasons": non_actionable_reasons,
        "avg_prob_buy": avg_prob,
    }


def analyse_execution(rows: list[dict]) -> dict:
    total = len(rows)
    by_decision: Counter = Counter()
    dry_runs = 0
    live_orders = 0
    rejected = 0

    for row in rows:
        decision = row.get("decision", "?")
        by_decision[decision] += 1
        if decision == "accepted_dry_run":
            dry_runs += 1
        elif decision == "sent_live":
            live_orders += 1
        elif decision in ("blocked_risk", "blocked_no_quote_balance",
                          "blocked_existing_position", "blocked_no_inventory",
                          "rejected_exchange", "ignored_base_not_enabled"):
            rejected += 1

    return {
        "total": total,
        "by_decision": by_decision,
        "dry_runs": dry_runs,
        "live_orders": live_orders,
        "rejected": rejected,
    }


def analyse_paper(rows: list[dict]) -> dict:
    total = len(rows)
    buy_fills = 0
    sell_fills = 0
    realized_pnl = 0.0
    last_equity: float | None = None
    last_cash: float | None = None
    last_drawdown: float | None = None
    fills_by_asset: Counter = Counter()
    pnl_by_asset: defaultdict[str, float] = defaultdict(float)
    holding_ms_list: list[float] = []
    open_positions_at_fill: list[int] = []
    by_decision: Counter = Counter()

    for row in rows:
        decision = row.get("decision", "?")
        by_decision[decision] += 1
        product_id = row.get("product_id", "?")

        if decision == "paper_buy_filled":
            buy_fills += 1
            fills_by_asset[product_id] += 1
            try:
                last_equity = float(row["equity"])
                last_cash = float(row["cash"])
                last_drawdown = float(row["drawdown_pct"])
            except Exception:
                pass

        elif decision == "paper_sell_filled":
            sell_fills += 1
            fills_by_asset[product_id] += 1
            try:
                pnl = float(row.get("realized_pnl", 0))
                realized_pnl += pnl
                pnl_by_asset[product_id] += pnl
            except Exception:
                pass
            try:
                last_equity = float(row["equity"])
                last_cash = float(row["cash"])
                last_drawdown = float(row["drawdown_pct"])
            except Exception:
                pass
            try:
                holding_ms_list.append(float(row["holding_ms"]))
            except Exception:
                pass

    avg_holding_min = (
        sum(holding_ms_list) / len(holding_ms_list) / 60_000
        if holding_ms_list else None
    )

    return {
        "total": total,
        "by_decision": by_decision,
        "buy_fills": buy_fills,
        "sell_fills": sell_fills,
        "realized_pnl": realized_pnl,
        "last_equity": last_equity,
        "last_cash": last_cash,
        "last_drawdown": last_drawdown,
        "fills_by_asset": fills_by_asset,
        "pnl_by_asset": dict(pnl_by_asset),
        "avg_holding_min": avg_holding_min,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Render
# ──────────────────────────────────────────────────────────────────────────────

def print_report(
    hours: float,
    sig: dict,
    exe: dict,
    paper: dict,
    tail_rows: list[dict],
) -> None:
    W = 70
    now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    print()
    print("=" * W)
    print(f"  DIAGNÓSTICO RENDER — últimas {hours:.0f} h  ({now_str})")
    print("=" * W)

    # ── Signals ──────────────────────────────────────────────────────────────
    print()
    print("  INFERENCE SIGNALS")
    print(_separator())
    s = sig
    print(f"  Total señales       : {s['total']:,}")
    print(f"  Accionables         : {s['actionable']:,}  ({_pct(s['actionable'], s['total'])})")
    print(f"  Prob_buy promedio   : {_f(s['avg_prob_buy'], 4)}")
    print()
    print("  Distribución por tipo:")
    for label, count in sorted(s["by_signal"].items(), key=lambda x: -x[1]):
        print(f"    {label:<10} {count:>6,}  ({_pct(count, s['total'])})")
    if s["non_actionable_reasons"]:
        print()
        print("  Razones non-actionable (top 5):")
        for reason, count in s["non_actionable_reasons"].most_common(5):
            print(f"    {reason:<40} {count:>5,}")
    if s["by_asset"]:
        print()
        print("  Assets con más señales (top 10):")
        for asset, count in s["by_asset"].most_common(10):
            print(f"    {asset:<16} {count:>6,}")

    # ── Execution (dry_run) ───────────────────────────────────────────────────
    print()
    print("  ORDER MANAGER (execution.events)")
    print(_separator())
    e = exe
    print(f"  Total eventos       : {e['total']:,}")
    print(f"  Dry-runs aceptados  : {e['dry_runs']:,}")
    print(f"  Órdenes live        : {e['live_orders']:,}")
    print(f"  Rechazadas/bloqueadas: {e['rejected']:,}")
    if e["by_decision"]:
        print()
        print("  Decisiones:")
        for decision, count in sorted(e["by_decision"].items(), key=lambda x: -x[1]):
            print(f"    {decision:<40} {count:>6,}")

    # ── Paper ─────────────────────────────────────────────────────────────────
    print()
    print("  PAPER TRADER (paper.execution.events)")
    print(_separator())
    p = paper
    print(f"  Total eventos       : {p['total']:,}")
    print(f"  BUY fills           : {p['buy_fills']:,}")
    print(f"  SELL fills          : {p['sell_fills']:,}")
    print(f"  Realized PnL        : ${_f(p['realized_pnl'])}  USD")
    if p["last_equity"] is not None:
        print(f"  Equity actual       : ${_f(p['last_equity'])}")
        print(f"  Cash actual         : ${_f(p['last_cash'])}")
        dd = p["last_drawdown"]
        print(f"  Drawdown            : {_f(dd * 100, 3)}%")
    if p["avg_holding_min"] is not None:
        print(f"  Holding promedio    : {_f(p['avg_holding_min'])} min")

    if p["fills_by_asset"]:
        print()
        print("  Fills por asset:")
        for asset, count in sorted(p["fills_by_asset"].items(), key=lambda x: -x[1]):
            pnl = p["pnl_by_asset"].get(asset, 0.0)
            pnl_str = f"  pnl=${_f(pnl)}" if pnl != 0.0 else ""
            print(f"    {asset:<16} {count:>4,} fills{pnl_str}")

    if p["by_decision"]:
        print()
        print("  Decisiones paper (todos):")
        for decision, count in sorted(p["by_decision"].items(), key=lambda x: -x[1]):
            print(f"    {decision:<40} {count:>6,}")

    # ── Tail de señales recientes ─────────────────────────────────────────────
    if tail_rows:
        print()
        print(f"  ÚLTIMAS {len(tail_rows)} SEÑALES ACCIONABLES")
        print(_separator())
        fmt = "  {:<16} {:<6} {:>6} {:<8} {}"
        print(fmt.format("ASSET", "SIGNAL", "PROB", "ACTION", "DECISION/REASON"))
        print("  " + "─" * 66)
        for row in tail_rows:
            asset = row.get("product_id", "?")[:16]
            signal = row.get("signal", "?")[:6]
            prob = _f(row.get("prob_buy", "?"), 4)
            actionable = "YES" if str(row.get("actionable", "")).lower() == "true" else "no"
            reason = (row.get("reason", "") or row.get("decision", "") or "")[:30]
            print(fmt.format(asset, signal, prob, actionable, reason))

    print()
    print("=" * W)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico Redis del bot en Render")
    parser.add_argument("--hours", type=float, default=4.0, help="Horas hacia atrás a analizar (default: 4)")
    parser.add_argument("--tail", type=int, default=20, help="Nº últimas señales a mostrar en detalle (default: 20)")
    parser.add_argument("--redis-url", default=None, help="Redis URL override (default: env REDIS_URL)")
    args = parser.parse_args()

    redis_url = args.redis_url or os.getenv("REDIS_URL", "")
    if not redis_url:
        print("[ERROR] REDIS_URL no configurada. Exporta la variable o usa --redis-url.")
        print("        Ejemplo: REDIS_URL=rediss://user:pass@host:port python scripts/diagnose_render.py")
        sys.exit(1)

    print(f"\nConectando a Redis... ({redis_url[:40]}...)" if len(redis_url) > 40 else f"\nConectando a Redis... ({redis_url})")
    r = _build_redis(redis_url)
    try:
        r.ping()
        print("  OK — Redis disponible.")
    except Exception as exc:
        print(f"  [ERROR] No se pudo conectar a Redis: {exc}")
        sys.exit(1)

    min_id = _min_stream_id(args.hours)
    print(f"  Leyendo últimas {args.hours:.0f} h (desde ID {min_id})...")

    signal_rows = _read_stream(r, settings.STREAM_INFERENCE_SIGNALS, min_id)
    exec_rows   = _read_stream(r, settings.STREAM_EXECUTION_EVENTS, min_id)
    paper_rows  = _read_stream(r, settings.STREAM_PAPER_EXECUTION_EVENTS, min_id)

    print(f"  Señales: {len(signal_rows):,}  |  Execution: {len(exec_rows):,}  |  Paper: {len(paper_rows):,}")

    sig_stats   = analyse_signals(signal_rows)
    exe_stats   = analyse_execution(exec_rows)
    paper_stats = analyse_paper(paper_rows)

    # Tail: últimas N señales accionables (o si no hay, las últimas N del stream)
    tail_signals = [row for row in signal_rows if str(row.get("actionable", "")).lower() == "true"]
    if not tail_signals:
        tail_signals = signal_rows
    tail_rows = tail_signals[-args.tail:]

    print_report(args.hours, sig_stats, exe_stats, paper_stats, tail_rows)


if __name__ == "__main__":
    main()
