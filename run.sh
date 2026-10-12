#!/usr/bin/env sh
# Arranque para Linux / macOS: crea el entorno, instala dependencias, descarga modelos y ejecuta.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Creando entorno virtual..."
    python3 -m venv .venv
fi
if [ ! -f .venv/.instalado ]; then
    echo "Instalando dependencias..."
    .venv/bin/python -m pip install -r requirements.txt
    touch .venv/.instalado
fi
if [ ! -f models/hand_landmarker.task ] || [ ! -f models/face_landmarker.task ]; then
    .venv/bin/python download_models.py
fi
exec .venv/bin/python app.py "$@"
