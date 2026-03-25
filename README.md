# XGBoost Trading System

Proyecto base para research, entrenamiento y ejecucion de un sistema de trading crypto con XGBoost sobre velas de 1 minuto.

## Que incluye

- Construccion de velas desde trades individuales.
- Collector de Coinbase con publicacion en Redis streams.
- Feature engineering reutilizable para training y live.
- Target builder con `NET_RETURN_THRESHOLD` y `TRIPLE_BARRIER`.
- Seleccion de features con SHAP, permutation importance y filtro de correlacion.
- Walk-forward validation con gap y test final intacto.
- Registry de modelos con promocion conservadora.
- Trainer periodico, motor de inferencia y guardian de riesgo.
- Registry multi-activo, ejecucion dry-run y cliente privado de Coinbase.
- Orquestador `main.py` para correr servicios en paralelo.

## Estructura

```text
config/settings.py         Configuracion central
data/candle_builder.py     Trades -> velas 1m
data/collector.py          Coinbase WS -> Redis
features/calculator.py     Features tecnicas
features/selector.py       SHAP + permutation + estabilidad
target/builder.py          Definicion de labels
validation/walk_forward.py Validacion temporal
model/registry.py          Historial y promocion
model/trainer.py           Reentrenamiento periodico
model/inference.py         Senales multi-activo BUY/SELL/HOLD
exchange/coinbase_client.py Cliente privado Coinbase
execution/order_manager.py Ejecucion spot controlada
risk/guardian.py           Checkpoints de riesgo
main.py                    Orquestador
pipeline.py                Pipeline offline sobre historico
```

## Instalacion

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso rapido

Pipeline offline sobre un DataFrame historico:

```python
from pipeline import FeaturePipeline
from tests.helpers import make_synthetic_candles

pipeline = FeaturePipeline()
result = pipeline.run(make_synthetic_candles(2000))
print(result.status, result.auc)
```

Arranque del sistema:

```powershell
python main.py
```

Preparado para Render:

```powershell
python .\scripts\render_start.py
```

Ese bootstrap:

- copia el seed de modelos desde `seed_models/multi_asset_paper_24` al storage persistente si el disco esta vacio;
- usa `runtime/` para el registry efectivo y para el SQLite historico del trainer;
- luego arranca `main.py`.
- soporta Redis tanto por `REDIS_HOST`/`REDIS_PORT` como por `REDIS_URL`.

Tests:

```powershell
python -m pytest tests/ -v
```

## Historico y reentrenamiento

Formato esperado del archivo historico:

- Requeridas: `open_time`, `open`, `high`, `low`, `close`, `volume`
- Opcionales: `product_id`, `close_time`, `trade_count`
- `open_time` puede venir como milisegundos, segundos Unix o datetime parseable

Validar un archivo antes de importarlo:

```powershell
python .\scripts\import_historical_candles.py --path .\mi_historico.csv --product-id BTC-USD --dry-run
```

Descargar historico real desde Coinbase:

```powershell
python .\scripts\fetch_coinbase_history.py --symbols BTC/USD ETH/USD --timeframe 1m --candles 2400 --output-dir data/historical
```

Importar velas a Redis:

```powershell
python .\scripts\import_historical_candles.py --path .\mi_historico.csv --product-id BTC-USD --clear-stream
```

Reentrenar una vez desde archivo:

```powershell
python .\scripts\retrain_from_historical.py --path .\mi_historico.csv --product-id BTC-USD
```

Politica actual de reentrenamiento:

- cada activo guarda su historico completo de velas en `data/trainer_candle_history.sqlite3`;
- cada ciclo entrena usando todo el historico disponible del activo;
- las velas recientes reciben mas peso mediante `sample_weight` exponencial;
- se intenta `training continuation` cuando el modelo activo y el set de features son compatibles;
- si continuation falla, el trainer reentrena desde cero sobre la misma base historica ponderada;
- el registry nunca promueve el candidato si no mejora al activo.

Backfill inicial del SQLite historico del trainer desde los CSV ya descargados:

```powershell
python .\scripts\bootstrap_trainer_history_store.py
```

Ese comando:

- carga el universo principal de `24` activos al SQLite del trainer;
- conserva futuras velas via sincronizacion incremental desde Redis;
- deja un reporte en `reports/trainer_history_bootstrap.json`.

Descargar y entrenar varios activos por separado:

```powershell
python .\scripts\train_multi_asset_separate.py --exchange binanceus --timeframe 1m --candles 259200 --batch-size 1000 --output-dir data/historical_6m_binanceus --registry-root models/multi_asset_6m_binanceus
```

El material historico de fases anteriores y experimentos ya no forma parte del flujo principal.
Quedo archivado en `tests/legacy/`.

Screening del universo spot de Coinbase USA para priorizar activos entrenables:

```powershell
python .\scripts\screen_coinbase_operable_assets.py --top 30
```

Ese comando deja un JSON completo y un Markdown resumido en `reports/` con:

- filtros hard por estado, flags de trading, volumen 24h, edad del listing y cobertura reciente de velas 1m;
- score heuristico por liquidez, microestructura, estabilidad y madurez;
- tiers `train_now`, `observe` y `skip`.

Comparar ventana reciente vs 12 meses completos vs 12 meses ponderados:

```powershell
python .\scripts\compare_history_windows.py --exchange binanceus --timeframe 1m --report-path reports\history_window_comparison_12m_merged.json
```

Clonar el root live, aplicar ganadores historicos y dejar listo el registry nuevo:

```powershell
python .\scripts\rollout_live_registry.py --source-root models\multi_asset_6m_binanceus --target-root models\multi_asset_live_v2 --selection BTC/USDT=12m_flat --selection ETH/USDT=12m_weighted
```

Smoke test end-to-end sin Redis externo:

```powershell
python .\scripts\smoke_e2e.py
```

Probe live con feed real de Coinbase y Redis en memoria:

```powershell
$env:PRODUCTS_CSV='BTC-USD,ETH-USD'
$env:LIVE_PROBE_SECONDS='45'
python .\scripts\live_probe.py
```

Probe live multi-activo con auth privada Coinbase y ejecucion en `dry-run`:

```powershell
$env:LIVE_PROBE_SECONDS='45'
python .\scripts\live_multi_asset_probe.py
```

Collector hibrido de Coinbase:

- `COINBASE_WS_AUTH_ENABLED=true` autentica tambien las suscripciones de market data con JWT CDP.
- `COINBASE_CANDLE_RECONCILE_ENABLED=true` activa reconciliacion de velas 1m por REST tras gaps o reconexiones.
- `COINBASE_CANDLE_RECONCILE_LOOKBACK_MINUTES=3` define cuantas velas recientes revisar en cada reconciliacion.

Si quieres repartir los productos en varias conexiones websocket, puedes fijar:

```powershell
$env:COINBASE_WS_PRODUCTS_PER_CONNECTION='1'
```

Paper trading controlado sobre señales reales:

```powershell
$env:PAPER_TRADING_ENABLED='true'
python .\scripts\paper_trading_probe.py
```

Consolidar los `24` activos estudiados en un root paper separado, manteniendo `ZEC` fuera:

```powershell
python .\scripts\bootstrap_studied_universe_paper_root.py --target-root models/multi_asset_paper_24
```

Sesion paper unificada para esos `24` activos con datos reales y fills simulados:

```powershell
python .\scripts\run_studied_universe_paper_session.py --seconds 3600
```

Arranque del sistema completo con datos reales, reentrenamiento periodico, `dry-run` y paper trading para esos `24` activos:

```powershell
python .\scripts\run_studied_universe_live_paper.py
```

## GitHub y Render

El repo esta pensado para subir solo codigo y artefactos minimos de arranque:

- `seed_models/multi_asset_paper_24` queda versionado como seed para Render;
- el historico pesado, sesiones, SQLite local, logs y reportes quedan fuera via `.gitignore`;
- `render.yaml` arranca un worker en modo `dry-run + paper trading`, con Redis gestionado y storage persistente en `runtime/`.
- el deploy actual tambien incluye un `chop guard` para detectar activos en serrucho, subir el umbral de entrada, bajar el notional y enfriar temporalmente el activo tras stop-outs repetidos.

Importante:

- el SQLite historico grande del trainer no se sube a GitHub;
- en Render, el trainer arrancara con el seed de modelos y luego ira acumulando velas nuevas en el storage persistente;
- si despues quieres reentrenamiento con historico completo desde el primer dia, conviene sembrar ese SQLite por un canal externo al repo.

Las herramientas historicas de `phase1`, `phase2` y `ZEC` quedaron archivadas en `tests/legacy/`.

Redis real en WSL:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\redis-wsl.ps1 start
powershell -ExecutionPolicy Bypass -File .\scripts\redis-wsl.ps1 ping
powershell -ExecutionPolicy Bypass -File .\scripts\redis-wsl.ps1 status
```

## Estado actual

- Suite local: `62 passed`.
- Validacion temporal implementada con reserva temprana del test final.
- Communication bridge con Codex Cloud y Claude Code disponible en este workspace.
- `execution.events` publica decisiones auditables del ejecutor.
- Experimento `6m vs 12m` consolidado en `reports/history_window_comparison_12m_merged.json`.
- Root live por defecto: `models/multi_asset_live_v2`.
- Universo principal unificado: `24` activos estudiados en `models/multi_asset_paper_24`.
- `ZEC` queda fuera de la fase principal y archivado en `tests/legacy/`.
- Collector Coinbase endurecido con secuencia por conexion, auth opcional de market data y reconciliacion REST de velas 1m.
- Paper trader separado disponible en `paper.execution.events`.
- Screener automatico de operabilidad Coinbase disponible en `scripts/screen_coinbase_operable_assets.py`.
- `bootstrap_studied_universe_paper_root.py` consolida los `24` activos estudiados en un root paper separado.
- Ese root paper activa modelos promovidos y, cuando no existia promocion, el ultimo candidato disponible solo para evaluacion paper.
- `run_studied_universe_paper_session.py` permite evaluarlos juntos en paper con datos reales sin tocar el live principal.
- `run_studied_universe_live_paper.py` arranca `main.py` con esos `24` activos, paper trading activado y `ZEC` fuera de la cohorte principal.
- Scripts, datos, modelos y reportes de fases anteriores quedaron archivados dentro de `tests/legacy/`.
- El trainer persiste velas por activo en SQLite y reentrena con todo el historico ponderando mas lo reciente.
- `bootstrap_trainer_history_store.py` permite poblar el SQLite historico del trainer con el universo principal antes de arrancar sesiones largas.

## Herramientas de colaboracion

Codex Cloud:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\codex-collab.ps1 submit -Mode build -Prompt "Implementa la tarea"
```

Primer uso en una copia nueva del workspace:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\codex-collab.ps1 submit -Mode build -EnvironmentId "<tu_environment_id>" -Prompt "Implementa la tarea"
```

Ese primer `-EnvironmentId` queda guardado automaticamente en `.workspace_agents/codex_cloud_bridge.json` para las siguientes corridas.

Claude Code en VS Code:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\claude-vscode.ps1 ask -Prompt "Revisa este enfoque"
```
