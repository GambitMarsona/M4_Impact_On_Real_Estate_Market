from __future__ import annotations

from typing import Callable, Optional, Tuple
from pathlib import Path
from collections import deque
import os
import time
import re

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from sklearn.neighbors import BallTree

from core import Bundle, Pipeline, StepExecutionError


# -------------------------------------------
# Domyślne API: LocationIQ
# -------------------------------------------
def _default_geocoder(address: str, sleep_s: float = 0.5) -> Tuple[float, float]:
    """
    Domyślne geokodowanie przez LocationIQ.
    Zwraca (lat, lon) lub (nan, nan), jeśli się nie uda.
    """
    api_key = os.getenv("LOCATIONIQ_API_KEY")
    if not api_key:
        raise StepExecutionError(
            "Brak LOCATIONIQ_API_KEY w zmiennych środowiskowych. "
            "Ustaw go albo przekaż własny geocode_func do register_feature_engineering()."
        )

    addr = (address or "").strip()
    if not addr:
        return np.nan, np.nan

    url = "https://eu1.locationiq.com/v1/search"
    params = {
        "key": api_key,
        "q": addr,
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": "sgh-licencjat-otodom/1.0"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return np.nan, np.nan
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
    except Exception:
        lat, lon = np.nan, np.nan

    if sleep_s:
        time.sleep(sleep_s)
    return lat, lon


def _save_snapshot(
    df: pd.DataFrame,
    *,
    snapshot_dir: str | Path,
    snapshot_prefix: str,
) -> Path:
    """
    Zapisuje aktualny df do pliku CSV w katalogu snapshot_dir.
    Nazwa: <prefix><liczba_uzupelnionych_lat>.csv
    np. otodom_offers_api_5000.csv
    """
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Jeśli nie ma kolumny lat - liczy po prostu wiersze
    if "Szerokosc_geo" in df.columns:
        filled_count = int(df["Szerokosc_geo"].notna().sum())
    else:
        filled_count = len(df)

    fname = f"{snapshot_prefix}{filled_count}.csv"
    path = snapshot_dir / fname

    df.to_csv(path, index=False)
    return path


# ===========================================
# KROK 1: geokodowanie
# ===========================================
def register_feature_engineering(
    pipe: Pipeline,
    *,
    bundle: Bundle,
    address_col: str = "Adres",
    city_default: str = "Warszawa",
    geocode_func: Optional[Callable[[str], Tuple[float, float]]] = None,
    max_consecutive_failures: int = 1000,
    snapshot_dir: str | Path = "data",
    snapshot_prefix: str = "otodom_offers_api_",
    log_every: int = 0,         # co ile geokodowanych rekordów wypisać log (0 = brak logów)
    enable_geocode: bool = True # jeśli False → w ogóle nie dzwonimy do API, idziemy „dalej”
) -> Pipeline:
    """
    1 krok pipeline: geokodowanie adresów do lat/lon (opcjonalnie wyłączalne).
    """
    if geocode_func is None:
        geocode_func = _default_geocoder

    @pipe.register(
        name="fe_geocode",
        on_table="data",
        requires=(),
        produces=("Szerokosc_geo", "Dlugosc_geo"),
        description=f"Geokodowanie kolumny '{address_col}' do lat/lon.",
        skip_if_all_produced_present=False,
    )
    def _fe_geocode(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        if address_col not in df.columns:
            raise StepExecutionError(
                f"fe_geocode: brak kolumny adresu '{address_col}' w data."
            )

        # zapewnij kolumny Szerokosc_geo/Dlugosc_geo
        if "Szerokosc_geo" not in df.columns:
            df["Szerokosc_geo"] = np.nan
        else:
            df["Szerokosc_geo"] = pd.to_numeric(df["Szerokosc_geo"], errors="coerce")

        if "Dlugosc_geo" not in df.columns:
            df["Dlugosc_geo"] = np.nan
        else:
            df["Dlugosc_geo"] = pd.to_numeric(df["Dlugosc_geo"], errors="coerce")

        # flaga próby geokodowania
        if "proba_geokodowania" not in df.columns:
            df["proba_geokodowania"] = np.int8(0)
        else:
            df["proba_geokodowania"] = (
                pd.to_numeric(df["proba_geokodowania"], errors="coerce")
                .fillna(0)
                .astype("int8")
            )

        if not enable_geocode:
            bundle.set("data", df)
            return df

        addr = df[address_col].astype("string").fillna("").str.strip()
        need = (
            addr.ne("")
            & (df["Szerokosc_geo"].isna() | df["Dlugosc_geo"].isna())
            & (df["proba_geokodowania"] != 1)
        )
        idxs = df.index[need]

        if len(idxs) == 0:
            bundle.set("data", df)
            return df

        fail_streak = 0
        counter = 0
        recent_nan_flags = deque(maxlen=log_every if log_every and log_every > 0 else 1)
        total_to_process = len(idxs)

        try:
            for i in idxs:
                counter += 1
                a = addr.loc[i]
                if not a:
                    continue

                if city_default and city_default.lower() not in a.lower():
                    full_addr = f"{a}, {city_default}, Polska"
                else:
                    full_addr = a

                # oznaczenie, że była próba geokodowania
                df.at[i, "proba_geokodowania"] = np.int8(1)

                lat, lon = geocode_func(full_addr)

                df.at[i, "Szerokosc_geo"] = lat
                df.at[i, "Dlugosc_geo"] = lon

                is_nan_pair = bool(np.isnan(lat) and np.isnan(lon))

                # logowanie co N
                if log_every and log_every > 0:
                    recent_nan_flags.append(is_nan_pair)
                    if counter % log_every == 0:
                        nan_recent = sum(1 for x in recent_nan_flags if x)
                        window_size = min(log_every, len(recent_nan_flags))
                        print(
                            f"[FE] Geocoded {counter}/{total_to_process} rows. "
                            f"NaN w ostatnich {window_size}: {nan_recent}."
                        )

                # obsługa fail_streak
                if is_nan_pair:
                    fail_streak += 1
                    if fail_streak >= max_consecutive_failures:
                        bundle.set("data", df)
                        snap_path = _save_snapshot(
                            df,
                            snapshot_dir=snapshot_dir,
                            snapshot_prefix=snapshot_prefix,
                        )
                        raise StepExecutionError(
                            "fe_geocode: geokodowanie przerwane – API prawdopodobnie przestało działać "
                            f"({fail_streak} kolejnych odpowiedzi (NaN, NaN)). "
                            f"Stan danych zapisany w bundle['data'] oraz w pliku: {snap_path}."
                        )
                else:
                    fail_streak = 0

        except KeyboardInterrupt:
            # Użytkownik przerwał ręcznie (kernel interrupt)
            bundle.set("data", df)
            snap_path = _save_snapshot(
                df,
                snapshot_dir=snapshot_dir,
                snapshot_prefix=snapshot_prefix,
            )
            raise StepExecutionError(
                "fe_geocode: przerwane przez użytkownika (KeyboardInterrupt). "
                f"Stan danych zapisany w bundle['data'] oraz w pliku: {snap_path}."
            )

        # pełne przejście bez fail_streak-limit – normalny koniec
        bundle.set("data", df)
        snap_path = _save_snapshot(
            df,
            snapshot_dir=snapshot_dir,
            snapshot_prefix=snapshot_prefix,
        )
        print(f"[FE] Pełne geokodowanie zakończone, zapisano snapshot: {snap_path}")
        return df

    return pipe


# ===========================================
# KROK 2: cechy przestrzenne (BallTree + dzielnice + cena/m2)
# ===========================================
def _clean_price_to_float(x: str | float | int) -> float:
    """
    Czyści tekstową cenę typu '1 234 567 zł' → 1234567.0
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    s = re.sub(r"[^0-9,\.]", "", s)
    s = s.replace(",", ".")
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _pick_first_existing(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise StepExecutionError(
        f"Nie znalazłem żadnej z kolumn {candidates} w data. Dostępne kolumny: {list(df.columns)}"
    )


def _prepare_latlon_from_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Upewnia się, że gdf ma kolumny lat/lon (EPSG:4326).
    """
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    if "lat" not in gdf.columns or "lon" not in gdf.columns:
        gdf["lat"] = gdf.geometry.y
        gdf["lon"] = gdf.geometry.x
    return gdf


def _compute_balltree_distance_m(
    base_df: pd.DataFrame,
    valid_mask: pd.Series,
    poi_latlon: np.ndarray,
    earth_radius_m: float = 6371000.0,
) -> np.ndarray:
    """
    Zwraca wektor o długości len(base_df) z odległościami w metrach (NaN tam,
    gdzie brak lat/lon lub brak punktów referencyjnych).
    """
    n = len(base_df)
    out = np.full(n, np.nan, dtype=float)

    if poi_latlon.size == 0:
        return out  

    base_coords = base_df.loc[valid_mask, ["Szerokosc_geo", "Dlugosc_geo"]].to_numpy()
    if base_coords.size == 0:
        return out  
    
    base_rad = np.radians(base_coords)
    poi_rad = np.radians(poi_latlon)

    tree = BallTree(poi_rad, metric="haversine")
    dist_rad, _ = tree.query(base_rad, k=1)
    dist_m = dist_rad[:, 0] * earth_radius_m

    out[valid_mask.to_numpy()] = dist_m
    return out


def register_spatial_features(
    pipe: Pipeline,
    *,
    bundle: Bundle,
    geojson_dir: str | Path = "geojson",
    snapshot_dir: str | Path = "data",
    snapshot_prefix: str = "otodom_offers_api_",
    log_every: int = 0,
    green_filename: str = "green.geojson",
    metro_filename: str = "metro.geojson",
    supermarket_filename: str = "supermarket.geojson",
    districts_filename: str = "warsaw_districts.geojson",
) -> Pipeline:
    """
    Krok 2: cechy przestrzenne:
    - odległość do M1, M2, terenów zielonych, supermarketów (BallTree, metry),
    - przypisanie dzielnicy (warsaw_districts.geojson),
    - cena za m2 (float, z prostym przeliczeniem walut).
    """
    @pipe.register(
        name="spatial_features",
        on_table="data",
        requires=(),  
        produces=(
            "Odleglosc_metro_m1",
            "Odleglosc_metro_m2",
            "Odleglosc_tereny_zielone",
            "Odleglosc_supermarket",
            "Dzielnica",
            "Cena_za_m2",
        ),
        description="Cechy przestrzenne + dzielnica + cena/m2.",
        skip_if_all_produced_present=False,  
    )
    def _spatial_features(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df["Szerokosc_geo"] = pd.to_numeric(df.get("Szerokosc_geo"), errors="coerce")
        df["Dlugosc_geo"] = pd.to_numeric(df.get("Dlugosc_geo"), errors="coerce")
        valid = df["Szerokosc_geo"].notna() & df["Dlugosc_geo"].notna()

        if log_every:
            print(f"[SPATIAL] Rekordów z poprawnymi współrzędnymi: {valid.sum()} / {len(df)}")

        # -------------------------------------------
        # 1) wczytanie geojsonów
        # -------------------------------------------
        base_dir = Path(geojson_dir)

        green_path = base_dir / green_filename
        metro_path = base_dir / metro_filename
        supermarket_path = base_dir / supermarket_filename
        districts_path = base_dir / districts_filename

        if log_every:
            print(f"[SPATIAL] Wczytuję geojsony z: {base_dir}")

        if not green_path.is_file():
            raise StepExecutionError(f"Nie znaleziono pliku: {green_path}")
        if not metro_path.is_file():
            raise StepExecutionError(f"Nie znaleziono pliku: {metro_path}")
        if not supermarket_path.is_file():
            raise StepExecutionError(f"Nie znaleziono pliku: {supermarket_path}")
        if not districts_path.is_file():
            raise StepExecutionError(f"Nie znaleziono pliku: {districts_path}")

        green = gpd.read_file(green_path)
        metro = gpd.read_file(metro_path)
        supermarket = gpd.read_file(supermarket_path)
        districts = gpd.read_file(districts_path)

        green = _prepare_latlon_from_geometry(green)
        metro = _prepare_latlon_from_geometry(metro)
        supermarket = _prepare_latlon_from_geometry(supermarket)

        # -------------------------------------------
        # 2) podział metra na M1 vs M2
        # -------------------------------------------
        metro_line_col = None
        for c in ["colour"]:
            if c in metro.columns:
                metro_line_col = c
                break
        if metro_line_col is None:
            raise StepExecutionError(
                f"metro.geojson nie ma kolumny z linią metra (szukano: name). "
                f"Dostępne kolumny: {list(metro.columns)}"
            )

        line_vals = metro[metro_line_col].astype(str).str.lower()

        m1_mask = line_vals.str.contains("m1") | line_vals.str.contains("blue") | line_vals.str.contains("both")
        m2_mask = line_vals.str.contains("m2") | line_vals.str.contains("red") | line_vals.str.contains("both")

        metro_m1 = metro.loc[m1_mask]
        metro_m2 = metro.loc[m2_mask]

        # -------------------------------------------
        # 3) dystanse BallTree (metry)
        # -------------------------------------------
        if log_every:
            print("[SPATIAL] Liczę odległości BallTree...")

        metro_m1_latlon = metro_m1[["lat", "lon"]].to_numpy() if len(metro_m1) else np.empty((0, 2))
        metro_m2_latlon = metro_m2[["lat", "lon"]].to_numpy() if len(metro_m2) else np.empty((0, 2))
        green_latlon = green[["lat", "lon"]].to_numpy() if len(green) else np.empty((0, 2))
        supermarket_latlon = supermarket[["lat", "lon"]].to_numpy() if len(supermarket) else np.empty((0, 2))

        df["Odleglosc_metro_m1"] = _compute_balltree_distance_m(df, valid, metro_m1_latlon)
        df["Odleglosc_metro_m2"] = _compute_balltree_distance_m(df, valid, metro_m2_latlon)
        df["Odleglosc_tereny_zielone"] = _compute_balltree_distance_m(df, valid, green_latlon)
        df["Odleglosc_supermarket"] = _compute_balltree_distance_m(df, valid, supermarket_latlon)

        # -------------------------------------------
        # 4) dzielnica (spatial join)
        # -------------------------------------------
        if log_every:
            print("[SPATIAL] Spatial join z dzielnicami...")

        offers_gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df["Dlugosc_geo"], df["Szerokosc_geo"]),
            crs="EPSG:4326",
        )

        if districts.crs is None:
            districts = districts.set_crs(epsg=4326)
        else:
            districts = districts.to_crs(epsg=4326)

        district_name_col = None
        for c in ["district", "dzielnica", "name", "nazwa", "jpt_nazwa_", "JPT_NAZWA_"]:
            if c in districts.columns:
                district_name_col = c
                break

        if district_name_col is None:
            print("[SPATIAL] Uwaga: nie znaleziono kolumny z nazwą dzielnicy w warsaw_districts.geojson.")
            df["Dzielnica"] = np.nan
        else:
            joined = gpd.sjoin(
                offers_gdf,
                districts[[district_name_col, "geometry"]],
                how="left",
                predicate="within",
            )
            # mamy potencjalne duplikaty indeksów 
            district_series = joined[district_name_col]
            # bierzemy pierwszą dzielnicę dla danego indeksu oferty
            district_series = district_series.groupby(level=0).first()
            # dopasowanie do indeksu df
            df["Dzielnica"] = district_series.reindex(df.index)

        # -------------------------------------------
        # 5) cena za m2 (z przeliczeniem walut)
        # -------------------------------------------
        if log_every:
            print("[SPATIAL] Liczę cenę za m2...")

        price_col = _pick_first_existing(df, ["price", "Cena", "cena", "price_value"])
        area_col = _pick_first_existing(df, ["area", "Powierzchnia", "Powierzchnia_m2", "size", "metraz", "metraż"])
        currency_col = None
        for c in ["currency", "Waluta", "waluta", "price_currency"]:
            if c in df.columns:
                currency_col = c
                break

        prices_raw = df[price_col].apply(_clean_price_to_float)

        if currency_col is not None:
            curr = df[currency_col].astype(str).str.upper().str.strip()
        else:
            curr = pd.Series(["PLN"] * len(df), index=df.index)

        eur_to_pln = 4.3
        usd_to_pln = 4.0

        price_pln = prices_raw.copy()
        price_pln[curr.eq("EUR")] = price_pln[curr.eq("EUR")] * eur_to_pln
        price_pln[curr.eq("USD")] = price_pln[curr.eq("USD")] * usd_to_pln

        df["Cena"] = price_pln

        area_val = pd.to_numeric(df[area_col], errors="coerce")
        df["Cena_za_m2"] = df["Cena"] / area_val.replace(0, np.nan)

        cols_to_drop = []
        for c in [
            "dist_metro_m1_m",
            "dist_metro_m2_m",
            "dist_green_m",
            "dist_supermarket_m",
            "district",
            "lon",
            "lat",
            "price_pln",
            "price_per_m2",
            "price_sqm_pln",
        ]:
            if c in df.columns:
                cols_to_drop.append(c)
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # -------------------------------------------
        # 6) snapshot na koniec
        # -------------------------------------------
        bundle.set("data", df)
        snap_path = _save_snapshot(
            df,
            snapshot_dir=snapshot_dir,
            snapshot_prefix=snapshot_prefix,
        )
        print(f"[SPATIAL] Cechy przestrzenne policzone, zapisano snapshot: {snap_path}")

        return df

    return pipe



# ===========================================
# KROK 3: binarizacja "Informacje dodatkowe"
# ===========================================
def register_additional_info_binarization(
    pipe: Pipeline,
    *,
    bundle: Bundle,
    snapshot_dir: str | Path = "data",
    snapshot_prefix: str = "otodom_offers_api_",
    log_every: int = 0,
    info_col_candidates = ("Informacje dodatkowe", "informacje dodatkowe", "additional_info"),
    min_freq: int = 1,  
) -> Pipeline:
    """
    Krok 3: one-hot encoding kolumny 'Informacje dodatkowe'.
    Wyjściowe kolumny mają nazwy zbliżone do oryginalnych kategorii,
    z transliteracją polskich znaków i bez prefiksu 'info_'.
    """

    def _sanitize_dummy_col(name: str) -> str:
        name = str(name).strip().lower()
        translit_map = {
            "ą": "a",
            "ć": "c",
            "ę": "e",
            "ł": "l",
            "ń": "n",
            "ó": "o",
            "ś": "s",
            "ż": "z",
            "ź": "z",
        }
        for src, tgt in translit_map.items():
            name = name.replace(src, tgt)

        name = re.sub(r"\s+", "_", name)

        name = re.sub(r"[^0-9a-z_]", "", name)

        name = name.strip("_")

        if not name:
            name = "unknown"

        return name

    @pipe.register(
        name="additional_info_binarization",
        on_table="data",
        requires=(),         
        produces=(),          
        description="One-hot encoding kolumny 'Informacje dodatkowe'.",
        skip_if_all_produced_present=False, 
    )
    def _additional_info_binarization(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        info_col = _pick_first_existing(df, info_col_candidates)

        s = (
            df[info_col]
            .fillna("")
            .astype("string")
        )

        if log_every:
            print(f"[BIN] Start binarizacji kolumny '{info_col}'")
        
        # normalizacja separatorów: ; | / • + -> przecinki
        s_norm = s.str.lower()
        for sep in [";", "|", "/", "•", "+", "\n", "\t"]:
            s_norm = s_norm.str.replace(sep, ",", regex=False)
        
        # ujednolicenie spacji wokół przecinków
        s_norm = s_norm.str.replace(r"\s*,\s*", ",", regex=True)
        
        # obcięcie zbędnych przecinków na brzegach
        s_norm = s_norm.str.strip(" ,")

        # one-hot encoding po przecinku
        dummies = s_norm.str.get_dummies(sep=",")

        # wyrzucamy ewentualną pustą kolumnę
        if "" in dummies.columns:
            dummies = dummies.drop(columns=[""])

        if dummies.shape[1] == 0:
            if log_every:
                print("[BIN] Brak sensownych kategorii do binarizacji – nic nie dodano.")
            bundle.set("data", df)
            return df

        # opcjonalne filtrowanie rzadkich kategorii
        if min_freq > 1:
            freq = dummies.sum(axis=0)
            keep_cols = freq[freq >= min_freq].index
            dummies = dummies[keep_cols]
            if log_every:
                print(f"[BIN] Po odfiltrowaniu rzadkich cech (min_freq={min_freq}) zostało {len(keep_cols)} kolumn.")

        if dummies.shape[1] == 0:
            if log_every:
                print("[BIN] Po filtracji rzadkich cech nie zostało żadnych kolumn – nic nie dodano.")
            bundle.set("data", df)
            return df

        # sanitizacja nazw kolumn + sprawdzenie konfliktów
        new_cols = {}
        for raw_name in dummies.columns:
            col_name = _sanitize_dummy_col(raw_name)
            if col_name == "brak_danych":
                continue
            if col_name == "windy":
                continue
            if col_name in df.columns:
                if col_name == "winda":
                    continue
                raise StepExecutionError(
                    f"additional_info_binarization: kolumna docelowa '{col_name}' już istnieje w data. "
                    f"Źródłowa kategoria: '{raw_name}'."
                )
            new_cols[col_name] = dummies[raw_name].astype("int8")

        # podpięcie do df
        for col_name, series in new_cols.items():
            df[col_name] = series

        # usunięcia zgodnie z listą
        cols_to_drop = []
        for c in ["brak_danych", "Winda", "windy"]:
            if c in df.columns:
                cols_to_drop.append(c)
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        if log_every:
            print(f"[BIN] Dodano {len(new_cols)} binarnych kolumn z '{info_col}'.")

        # snapshot na koniec
        bundle.set("data", df)
        snap_path = _save_snapshot(
            df,
            snapshot_dir=snapshot_dir,
            snapshot_prefix=snapshot_prefix,
        )
        if log_every:
            print(f"[BIN] Snapshot po binarizacji zapisany: {snap_path}")

        return df

    return pipe