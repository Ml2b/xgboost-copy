"""Ajustes de entorno cargados automaticamente al iniciar Python."""

from __future__ import annotations

import os

# Evita warnings de joblib/loky en entornos Windows donde no se detectan cores fisicos.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))
