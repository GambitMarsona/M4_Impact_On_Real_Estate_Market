from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd

from core import Bundle, Pipeline, StepExecutionError
from feature_engineering import (
    register_feature_engineering,
    register_spatial_features,
    register_additional_info_binarization,  
)


DATA_DIR = Path("data")
SNAP_PREFIX = "otodom_offers_api_"  


def _find_best_api_file() -> Optional[Path]:
    """
    Szuka w katalogu data/ plików typu otodom_offers_api_XXXX.csv
    i zwraca ten z największym XXXX. Jeśli nie ma żadnego – zwraca None.
    """
    if not DATA_DIR.exists():
        return None

    best_path: Optional[Path] = None
    best_n = -1

    for p in DATA_DIR.glob(f"{SNAP_PREFIX}*.csv"):
        stem = p.stem  # np. "otodom_offers_api_5000"
        suffix = stem.replace(SNAP_PREFIX, "")
        try:
            n = int(suffix)
        except ValueError:
            continue
        if n > best_n:
            best_n = n
            best_path = p

    return best_path


def run_pipeline(
    data: str = "otodom_offers.csv",  
    log_every: int = 0,               
    geojson_dir: str | Path = "geojson",  
) -> Bundle:
    """
    Główna funkcja do użycia w notebooku.

    - jeśli w data/ istnieje plik otodom_offers_api_XXXX.csv:
        -> bierze ten z największym XXXX jako źródło
    - jeśli nie:
        -> bierze data/<data> (domyślnie data/otodom_offers.csv)

    Następnie:
    - sprawdza, czy są kompletne współrzędne (lat, lon bez NaN):
        -> jeśli TAK: geocode jest pomijany (enable_geocode=False),
        -> jeśli NIE: geocode próbuje uzupełnić braki (enable_geocode=True).
    - potem krok fe_spatial: dzielnice, cena/m2, dystanse do M1, M2, green, supermarket.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    meta: dict = {}
    best_api = _find_best_api_file()

    if best_api is not None:
        src_path = best_api
    else:
        src_path = DATA_DIR / data

    if not src_path.is_file():
        raise FileNotFoundError(f"Nie znalazłem pliku wejściowego: {src_path}")

    df = pd.read_csv(src_path)

    # sprawdzamy, czy mamy komplet współrzędnych
    has_full_coords = (
        ("lat" in df.columns)
        and ("lon" in df.columns)
        and df["lat"].notna().all()
        and df["lon"].notna().all()
    )

    enable_geocode = not has_full_coords

    meta["source_file"] = str(src_path)
    meta["started_at"] = datetime.now().isoformat()
    meta["error"] = None
    meta["latest_snapshot"] = None
    meta["geocode_enabled"] = enable_geocode

    bundle = Bundle()
    bundle.set("data", df)
    bundle.set("meta", meta)

    pipe = Pipeline(bundle=bundle)

    # KROK 1: geokodowanie
    register_feature_engineering(
        pipe,
        bundle=bundle,
        snapshot_dir=DATA_DIR,
        snapshot_prefix=SNAP_PREFIX,
        log_every=log_every,
        enable_geocode=enable_geocode,
    )

    # KROK 2: dzielnice + cena/m2 + dystanse
    register_spatial_features(
        pipe,
        bundle=bundle,
        geojson_dir=geojson_dir,  
        snapshot_dir=DATA_DIR,
        snapshot_prefix=SNAP_PREFIX,
        log_every=log_every,
    )
    # KROK 3: binarizacja "Informacje dodatkowe"
    register_additional_info_binarization(
        pipe,
        bundle=bundle,
        snapshot_dir=DATA_DIR,
        snapshot_prefix=SNAP_PREFIX,
        log_every=log_every,
    )

    try:
        pipe.run() 
    except StepExecutionError as e:
        meta["error"] = str(e)

    meta["finished_at"] = datetime.now().isoformat()
    meta["log"] = pipe.log_as_dataframe()

    latest = _find_best_api_file()
    meta["latest_snapshot"] = str(latest) if latest is not None else None

    bundle.set("meta", meta)
    bundle.set("data", bundle.get("data"))

    return bundle
