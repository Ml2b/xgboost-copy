"""Bootstrap seguro para Render con modelos seed y almacenamiento persistente."""

from __future__ import annotations

import contextlib
import http.server
import os
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data.history_store import CandleHistoryStore
from exchange.coinbase_client import CoinbaseAdvancedTradeClient

SEED_MODEL_ROOT = ROOT / "seed_models" / "multi_asset_paper_24"
MODEL_LOCK_PATH = ROOT / "runtime" / ".seed_copy.lock"
HISTORY_LOCK_PATH = ROOT / "runtime" / ".history_seed.lock"
DEFAULT_HISTORY_TARGET_ROWS = 1500
DEFAULT_HISTORY_CHUNK_LIMIT = 350
DEFAULT_HISTORY_MAX_REQUESTS_PER_ASSET = 6


def _start_bootstrap_health_server() -> http.server.HTTPServer:
    """Arranca un servidor HTTP minimo para que Render detecte el puerto durante el bootstrap."""
    port = int(os.getenv("PORT", "10000"))

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b'{"status":"bootstrapping"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # noqa: ANN002
            pass  # silenciar logs del servidor de bootstrap

    server = http.server.HTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[render_start] bootstrap health server listening on port {port}", flush=True)
    return server


def main() -> None:
    """Prepara el entorno persistente y arranca el worker principal."""
    bootstrap_server = _start_bootstrap_health_server()
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
    print(f"[render_start] model_root={model_root}", flush=True)
    print(f"[render_start] history_db_path={history_db_path}", flush=True)
    ensure_seed_models(model_root)
    ensure_history_seed(history_db_path)
    # Cerrar el socket del bootstrap antes de exec para liberar el puerto 10000
    try:
        bootstrap_server.server_close()
    except Exception:
        pass
    print("[render_start] launching main.py", flush=True)
    os.execvpe(
        sys.executable,
        [sys.executable, str(ROOT / "main.py")],
        os.environ.copy(),
    )


def ensure_seed_models(target_root: Path) -> None:
    """Copia el seed de modelos una sola vez al storage persistente de Render."""
    if target_root.exists() and any(target_root.iterdir()):
        print("[render_start] seed models already present on disk", flush=True)
        return
    if not SEED_MODEL_ROOT.exists():
        raise FileNotFoundError(f"No existe el seed de modelos: {SEED_MODEL_ROOT}")

    MODEL_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = acquire_lock(MODEL_LOCK_PATH)
    try:
        if target_root.exists() and any(target_root.iterdir()):
            print("[render_start] seed models became available during lock wait", flush=True)
            return

        temp_target = target_root.parent / f"{target_root.name}.tmp"
        if temp_target.exists():
            shutil.rmtree(temp_target)
        print(f"[render_start] copying seed models from {SEED_MODEL_ROOT}", flush=True)
        shutil.copytree(SEED_MODEL_ROOT, temp_target)
        if target_root.exists():
            shutil.rmtree(target_root)
        temp_target.rename(target_root)
        print("[render_start] seed models copied", flush=True)
    finally:
        release_lock(MODEL_LOCK_PATH, lock_fd)


def ensure_history_seed(target_db_path: Path) -> None:
    """Si el historico esta vacio o corto, lo rellena con velas recientes de Coinbase."""
    if os.getenv("RENDER_HISTORY_BOOTSTRAP_ENABLED", "true").strip().lower() != "true":
        print("[render_start] history bootstrap disabled by env", flush=True)
        return

    product_ids = parse_product_ids_from_env(os.getenv("PRODUCTS_CSV", ""))
    if not product_ids:
        print("[render_start] no explicit PRODUCTS_CSV found for history bootstrap", flush=True)
        return

    target_rows = int(os.getenv("RENDER_HISTORY_BOOTSTRAP_TARGET_ROWS", str(DEFAULT_HISTORY_TARGET_ROWS)))
    chunk_limit = int(os.getenv("RENDER_HISTORY_BOOTSTRAP_CHUNK_LIMIT", str(DEFAULT_HISTORY_CHUNK_LIMIT)))
    max_requests = int(
        os.getenv(
            "RENDER_HISTORY_BOOTSTRAP_MAX_REQUESTS_PER_ASSET",
            str(DEFAULT_HISTORY_MAX_REQUESTS_PER_ASSET),
        )
    )

    lock_fd = acquire_lock(HISTORY_LOCK_PATH)
    try:
        store = CandleHistoryStore(db_path=target_db_path)
        pending_products = [
            product_id
            for product_id in product_ids
            if store.get_row_count_for_base(product_id.split("-", 1)[0]) < target_rows
        ]
        if not pending_products:
            print("[render_start] trainer history already seeded on disk", flush=True)
            return

        client = CoinbaseAdvancedTradeClient()
        print(
            f"[render_start] seeding trainer history for {len(pending_products)} products target_rows={target_rows}",
            flush=True,
        )
        seeded_assets = 0
        for product_id in pending_products:
            before_rows = store.get_row_count_for_base(product_id.split("-", 1)[0])
            try:
                row_count = seed_recent_history_for_product(
                    store=store,
                    client=client,
                    product_id=product_id,
                    target_rows=target_rows,
                    chunk_limit=chunk_limit,
                    max_requests=max_requests,
                )
            except Exception as exc:
                print(
                    f"[render_start] history seed failed product_id={product_id} error={exc}",
                    flush=True,
                )
                continue
            print(
                f"[render_start] history seeded product_id={product_id} rows_before={before_rows} rows_after={row_count}",
                flush=True,
            )
            seeded_assets += 1
        print(
            f"[render_start] trainer history bootstrap completed seeded_assets={seeded_assets}",
            flush=True,
        )
    finally:
        release_lock(HISTORY_LOCK_PATH, lock_fd)


def seed_recent_history_for_product(
    store: CandleHistoryStore,
    client: CoinbaseAdvancedTradeClient,
    product_id: str,
    target_rows: int,
    chunk_limit: int,
    max_requests: int,
) -> int:
    """Descarga velas recientes de un producto y las persiste en el SQLite del trainer."""
    base_asset = product_id.split("-", 1)[0].upper()
    current_rows = store.get_row_count_for_base(base_asset)
    if current_rows >= target_rows:
        return current_rows

    prefer_private = False
    end_seconds = int(time.time() // 60) * 60
    candles_by_open_time: dict[int, dict[str, object]] = {}
    requests_made = 0

    while len(candles_by_open_time) + current_rows < target_rows and requests_made < max_requests:
        start_seconds = max(0, end_seconds - (chunk_limit * 60))
        candles = client.get_candles(
            product_id=product_id,
            start=start_seconds,
            end=end_seconds,
            granularity="ONE_MINUTE",
            limit=chunk_limit,
            prefer_private=prefer_private,
        )
        if not candles:
            break

        for candle in candles:
            candles_by_open_time[int(candle["open_time"])] = candle

        earliest_open_ms = min(int(candle["open_time"]) for candle in candles)
        end_seconds = max(0, (earliest_open_ms // 1000) - 60)
        requests_made += 1
        time.sleep(0.15)

        if len(candles) < chunk_limit:
            break

    if candles_by_open_time:
        frame = pd.DataFrame(
            sorted(candles_by_open_time.values(), key=lambda item: int(item["open_time"]))
        )
        store.upsert_frame(
            frame,
            source_name=f"render_bootstrap:{product_id}",
            chunk_size=5000,
        )
    return store.get_row_count_for_base(base_asset)


def parse_product_ids_from_env(raw_products_csv: str) -> list[str]:
    """Normaliza PRODUCTS_CSV a una lista ordenada de product_ids."""
    product_ids: list[str] = []
    for token in raw_products_csv.split(","):
        normalized = token.strip().upper()
        if not normalized:
            continue
        if normalized not in product_ids:
            product_ids.append(normalized)
    return product_ids


def acquire_lock(lock_path: Path, timeout_seconds: int = 30) -> int:
    """Toma un lockfile simple para evitar doble copia del seed."""
    started = time.monotonic()
    while True:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            if time.monotonic() - started >= timeout_seconds:
                raise TimeoutError(f"No se pudo adquirir el lock de bootstrap en Render: {lock_path}")
            time.sleep(0.5)


def release_lock(lock_path: Path, lock_fd: int) -> None:
    """Libera el lockfile del bootstrap."""
    os.close(lock_fd)
    with contextlib.suppress(FileNotFoundError):
        lock_path.unlink()


if __name__ == "__main__":
    main()
