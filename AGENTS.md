# AGENTS.md

## Descripcion del proyecto

Sistema de trading crypto con XGBoost para senales BUY, SELL y HOLD.
Target principal: direccion futura del precio sobre velas de 1 minuto.
Stack principal: Python 3.11, XGBoost, Redis, Coinbase WebSocket API.

## Estructura del proyecto

```text
xgboost/
|- config/
|  \- settings.py
|- data/
|  |- collector.py
|  \- candle_builder.py
|- features/
|  |- calculator.py
|  |- pool.py
|  \- selector.py
|- target/
|  \- builder.py
|- validation/
|  \- walk_forward.py
|- model/
|  |- trainer.py
|  |- registry.py
|  \- inference.py
|- risk/
|  \- guardian.py
|- tests/
|- main.py
|- pipeline.py
|- requirements.txt
\- AGENTS.md
```

## Reglas de desarrollo

### Siempre

- Correr `python -m pytest tests/` despues de modificar modulos.
- Verificar que no haya data leakage.
- Comentar en espanol.
- Usar type hints en todas las funciones publicas.

### Nunca

- Mezclar train, validacion y test final.
- Usar el precio raw como feature directa fuera de transformaciones derivadas.
- Usar timestamp bruto como feature; solo codificacion ciclica.
- Reentrenar en cada tick.

### Convenciones de codigo

- Variables en `snake_case`.
- Clases en `PascalCase`.
- Constantes en mayusculas dentro de `config/settings.py`.
- Imports: stdlib, third party, local.

### Tests

- Cada modulo critico debe tener tests.
- Minimo: caso normal y caso edge.
- Ejecutar con `python -m pytest tests/ -v`.

## Reglas cuantitativas

- `FeatureCalculator` debe ser exactamente el mismo en training y en live.
- La seleccion de features ocurre dentro de cada fold.
- El test final se reserva antes de cualquier decision y se evalua una sola vez.
- Los umbrales de SHAP, permutation y correlacion son heuristicas y deben documentarse como tales.
- `ModelRegistry` nunca debe promover un modelo peor al activo.

## Flujo de colaboracion Codex local + Cloud

Cuando el usuario pida construir, revisar, debuggear, validar o evaluar:

1. Empezar localmente en este workspace.
2. Si `.workspace_agents/codex_cloud_bridge.json` tiene `defaultEnvironmentId`, lanzar una tarea paralela con `scripts/codex-collab.ps1`.
3. Usar `-Mode build` para implementacion, `-Mode review` para revision y `-Mode evaluate` para validacion funcional.
4. Seguir trabajando localmente mientras Cloud corre.
5. Revisar `status` y `diff` antes de aplicar cambios de Cloud.
6. No aplicar diffs de Cloud a ciegas.
7. Resumir en la respuesta final lo hecho localmente y cualquier aporte de Cloud.

## Claude Code en VS Code

Este workspace tambien puede consultar el bridge local de Claude Code con:

- `powershell -ExecutionPolicy Bypass -File .\scripts\claude-vscode.ps1 health`
- `powershell -ExecutionPolicy Bypass -File .\scripts\claude-vscode.ps1 ask -Prompt "Tu pregunta"`

