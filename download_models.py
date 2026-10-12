"""Descarga los modelos necesarios en la carpeta models/.

Cada modelo se descarga primero a un archivo .part y solo se renombra al terminar,
asi una descarga interrumpida no deja un modelo corrupto que parezca valido.
"""
import argparse
import sys
import urllib.request

from airsense.config import MODELS_DIR, MODEL_URLS

MIN_SIZE = 100_000   # bytes: cualquier modelo valido pesa varios MB


def _progress(name):
    def hook(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            pct = min(100.0, done / total * 100)
            print(f"\r   {name}: {pct:5.1f}%  ({done / 1e6:.1f}/{total / 1e6:.1f} MB)", end="", flush=True)
    return hook


def download(name, url, force=False):
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size > MIN_SIZE and not force:
        print(f"[ok] {name} ya existe")
        return True
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"[descargando] {name}")
    try:
        urllib.request.urlretrieve(url, part, reporthook=_progress(name))
        print()
        if part.stat().st_size < MIN_SIZE:
            raise RuntimeError(f"archivo demasiado pequeño ({part.stat().st_size} bytes)")
        part.replace(dest)
        print(f"   -> {dest.stat().st_size / 1e6:.1f} MB")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n   ERROR: {exc}\n   URL: {url}")
        part.unlink(missing_ok=True)
        return False


def main():
    ap = argparse.ArgumentParser(description="Descarga los modelos de MediaPipe en models/")
    ap.add_argument("--force", action="store_true", help="volver a descargar aunque ya existan")
    args = ap.parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ok = all([download(name, url, args.force) for name, url in MODEL_URLS.items()])
    if not ok:
        sys.exit(1)
    print("Modelos listos.")


if __name__ == "__main__":
    main()
