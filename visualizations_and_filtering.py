"""
visualizations_and_filtering.py

- Immutable filtering based on CFG["Filters"]
  Supported keys:
    - to_nan
    - drop_columns
    - drop_na
    - fill_na
    - impute_median
    - range

- Missing-data report before visualizations
- Visualization registry + runner: run_from_bundle(bundle, CFG)
- Auto-save to ./visualizations (creates dir if missing)

- Visualizations included:
    - warsaw_price_isolines (Ordinary Kriging, 50 samples per district)
    - violin_price_by_year_group
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Filtering (IMMUTABLE)
# ============================================================
def apply_filters(df: pd.DataFrame, Filters: Mapping[str, Any]) -> pd.DataFrame:
    """
    Correct and SAFE filter order.
    """
    out = df.copy()

    # 1) normalize textual missing values -> NaN
    to_nan = Filters.get("to_nan", [])
    if to_nan:
        out = out.replace(list(to_nan), np.nan)

    # 2) drop entire columns
    drop_columns = Filters.get("drop_columns", [])
    if drop_columns:
        cols = [c for c in drop_columns if c in out.columns]
        out = out.drop(columns=cols)

    # 3) fill NaN with constants (strings/categories)
    fill_na = Filters.get("fill_na", {})
    for col, val in fill_na.items():
        if col in out.columns:
            out[col] = out[col].fillna(val)

    # 3.5) map values (recode categories/strings -> numbers etc.)
    map_values = Filters.get("map_values", {})
    for col, mapping in map_values.items():
        if col in out.columns:
            out[col] = out[col].replace(mapping)

    # 4) explicit type casting  🔑🔑🔑
    astype = Filters.get("astype", {})
    for col, dtype in astype.items():
        if col not in out.columns:
            continue

        if dtype in ("float", "int"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if dtype == "int":
                out[col] = out[col].astype("Int64")
        else:
            out[col] = out[col].astype(dtype)

    # 5) impute median (now SAFE)
    for col in Filters.get("impute_median", []):
        if col in out.columns:
            med = out[col].median(skipna=True)
            if not pd.isna(med):
                out[col] = out[col].fillna(med)

    # 6) drop rows with NaN in required numeric columns
    drop_na = Filters.get("drop_na", [])
    if drop_na:
        out = out.dropna(subset=drop_na)

    # 7) numeric ranges (now SAFE)
    for col, (lo, hi) in Filters.get("range", {}).items():
        if col in out.columns:
            out = out[out[col].between(lo, hi)]

    return out




# ============================================================
# Missing-data report
# ============================================================

def missing_report(df: pd.DataFrame, top_n: int = 10) -> None:
    na_counts = df.isna().sum().sort_values(ascending=False)
    na_counts = na_counts[na_counts > 0]

    print("\n=== Missing data report (after Filters) ===")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")
    print(f"Columns with NaN: {len(na_counts)} | Total NaNs: {int(na_counts.sum()):,}")

    if not na_counts.empty:
        print("\nTop columns by missing values:")
        print(na_counts.head(top_n).to_string())


# ============================================================
# Stratified sampling
# ============================================================

def stratified_sample(
    df: pd.DataFrame,
    group_col: str,
    n_per_group: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Returns NEW df with up to n_per_group samples per group.
    """
    if group_col not in df.columns:
        raise KeyError(f"Missing group column '{group_col}' for stratified_sample")

    def _sampler(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) <= n_per_group:
            return g
        return g.sample(n=n_per_group, random_state=random_state)

    return (
        df.groupby(group_col, group_keys=False, dropna=False)
          .apply(_sampler)
          .reset_index(drop=True)
    )


# ============================================================
# Visualization registry
# ============================================================

_VIZ_REGISTRY: Dict[str, Callable[..., Tuple[Any, Any]]] = {}


def register_viz(name: str):
    def decorator(func: Callable[..., Tuple[Any, Any]]):
        _VIZ_REGISTRY[name] = func
        return func
    return decorator


def run_visualization(name: str, df: pd.DataFrame, **params):
    if name not in _VIZ_REGISTRY:
        raise KeyError(f"Visualization '{name}' is not registered")
    return _VIZ_REGISTRY[name](df, **params)


# ============================================================
# Auto-save
# ============================================================

def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure_default(fig, stem: str, out_dir: str | Path = "visualizations") -> Path:
    out_path = _ensure_dir(out_dir) / f"{stem}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    return out_path


# ============================================================
# Notebook runner
# ============================================================

def run_from_bundle(bundle, CFG: Mapping[str, Any]):
    """
    Runs all visualizations defined in CFG.
    """
    import matplotlib.pyplot as plt

    df_base = bundle.get("data")
    if df_base is None:
        raise KeyError("bundle.get('data') returned None")

    df_filtered = apply_filters(df_base, CFG.get("Filters", {}))

    # Report missingness before visualizations
    missing_report(df_filtered)

    results = []
    for item in CFG.get("visualizations", []):
        if not (isinstance(item, (tuple, list)) and len(item) == 3):
            raise ValueError(
                "Each CFG['visualizations'] item must be: (name, params_dict, stem)"
            )

        name, params, stem = item
        params = params or {}

        fig, ax = run_visualization(name, df_filtered, **params)
        saved_path = save_figure_default(fig, stem)

        plt.show()
        results.append((name, stem, saved_path, fig, ax))

    return results


# ============================================================
# Visualization: Warsaw price isolines (Kriging)
# ============================================================

@register_viz("warsaw_price_isolines")
def warsaw_price_isolines(
    df: pd.DataFrame,
    *,
    grid: int = 300,
    lon_col: str = "Dlugosc_geo",
    lat_col: str = "Szerokosc_geo",
    value_col: str = "Cena_za_m2",
    district_col: str = "Dzielnica",
    geojson_path: str = "geojson/warsaw_districts.geojson",
    levels: int = 10,
    cmap: str = "jet",
    title: str = "Mapa izolinii cenowych w Warszawie",
    cbar_label: str = "Cena za m² (PLN)",
    n_per_district: int = 50,
    seed: int = 42,
    annotate: bool = True,
):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    from pykrige.ok import OrdinaryKriging
    import numpy as np

    # --- walidacja
    for c in (lon_col, lat_col, value_col, district_col):
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' required by warsaw_price_isolines")

    # --- dane
    work = df[[lon_col, lat_col, value_col, district_col]].dropna().copy()
    work = stratified_sample(work, district_col, n_per_district, seed)

    if len(work) < 10:
        raise ValueError("Too few points for kriging")

    # --- granice Warszawy
    warsaw = gpd.read_file(geojson_path)
    if warsaw.crs is None:
        warsaw = warsaw.set_crs("EPSG:4326")

    warsaw_2180 = warsaw.to_crs(epsg=2180)
    warsaw_4326 = warsaw_2180.to_crs(epsg=4326)

    # --- punkty
    gdf = gpd.GeoDataFrame(
        work,
        geometry=gpd.points_from_xy(work[lon_col], work[lat_col]),
        crs="EPSG:4326",
    ).to_crs(epsg=2180)

    x, y, z = gdf.geometry.x.values, gdf.geometry.y.values, gdf[value_col].values

    # --- siatka
    minx, miny, maxx, maxy = warsaw_2180.total_bounds
    grid_x = np.linspace(minx, maxx, grid)
    grid_y = np.linspace(miny, maxy, grid)

    OK = OrdinaryKriging(x, y, z, variogram_model="spherical")
    zstar, _ = OK.execute("grid", grid_x, grid_y)

    GX, GY = np.meshgrid(grid_x, grid_y)
    grid_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(GX.ravel(), GY.ravel()),
        crs="EPSG:2180",
    )

    mask = grid_pts.within(warsaw_2180.unary_union).values.reshape(GX.shape)
    zstar = np.where(mask, zstar, np.nan)

    # --- do 4326 pod wykres
    grid_4326 = grid_pts.to_crs(epsg=4326)
    lons = grid_4326.geometry.x.values.reshape(GX.shape)
    lats = grid_4326.geometry.y.values.reshape(GX.shape)

    # --- rysowanie
    fig, ax = plt.subplots(figsize=(10, 8))

    cf = ax.contourf(lons, lats, zstar, levels=levels, cmap=cmap)
    ax.contour(lons, lats, zstar, levels=levels, colors="black", linewidths=0.6)

    warsaw_4326.boundary.plot(ax=ax, color="black", linewidth=1.2)

    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel("Długość geograficzna (°)")
    ax.set_ylabel("Szerokość geograficzna (°)")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.35)

    # ============================================================
    # ETYKIETY DZIELNIC (1:1 jak w Twoim kodzie)
    # ============================================================
    if annotate:
        warsaw_4326["centroid"] = warsaw_4326.geometry.centroid

        rotation_map = {
            "Wola": 20,
            "Praga Północ": -60,
            "Ochota": 20,
            "Wesoła": -20,
            "Bemowo": -20,
            "Śródmieście": -60,
            "Żoliborz": 20,
        }

        offset_map = {
            "Wola": (0.000, 0.000),
            "Praga Północ": (-0.005, -0.015),
            "Ochota": (0.000, -0.005),
            "Wesoła": (0.000, -0.003),
            "Bemowo": (0.000, 0.000),
            "Śródmieście": (-0.005, -0.015),
            "Żoliborz": (0.000, -0.005),
        }

        for _, row in warsaw_4326.iterrows():
            name = row.get("name", None)
            if name is None:
                continue

            cx, cy = row["centroid"].x, row["centroid"].y
            rot = rotation_map.get(name, 0)
            dx, dy = offset_map.get(name, (0.0, 0.0))

            txt = ax.annotate(
                name,
                xy=(cx + dx, cy + dy),
                ha="center",
                fontsize=10,
                color="black",
                rotation=rot,
                zorder=20,
            )


    return fig, ax


# ============================================================
# Visualization: Violin plot by year group
# ============================================================
@register_viz("violin_price_by_year_jenks")
def violin_price_by_year_jenks(
    df: pd.DataFrame,
    *,
    year_col: str = "Rok budowy",
    value_col: str = "Cena_za_m2",
    n_groups: int = 3,          # ← JEDYNY PARAMET W CFG
    palette: str = "hls",
    title: str = "Rozkład ceny za m² względem roku budowy (grupowanie Jenks)",
):
    """
    Violin plot with year grouping via Jenks Natural Breaks.
    Objective:
      - minimize within-group variance
      - maximize between-group mean differences
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import jenkspy

    # --- walidacja
    for c in (year_col, value_col):
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' required by violin_price_by_year_jenks")

    work = df[[year_col, value_col]].copy()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[year_col, value_col])

    if work[year_col].nunique() < n_groups:
        raise ValueError("Not enough unique year values for requested n_groups")

    # --- Jenks Natural Breaks
    breaks = jenkspy.jenks_breaks(work[year_col].values, n_classes=n_groups)
    # breaks: [min, b1, b2, ..., max]

    # --- labels
    labels = []
    for i in range(len(breaks) - 1):
        lo = int(round(breaks[i]))
        hi = int(round(breaks[i + 1]))
        if i == 0:
            labels.append(f"≤ {hi}")
        elif i == len(breaks) - 2:
            labels.append(f"> {lo}")
        else:
            labels.append(f"{lo}–{hi}")

    # --- assign groups
    work["rok_budowy_group"] = pd.cut(
        work[year_col],
        bins=breaks,
        labels=labels,
        include_lowest=True,
        ordered=True,
    )

    # --- wykres
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=work,
        x="rok_budowy_group",
        y=value_col,
        palette=palette,
        inner="box",
        cut=0,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Cena za m² (PLN)")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    return fig, ax


@register_viz("joyplot_price_by_finish_state")
def joyplot_price_by_finish_state(
    df: pd.DataFrame,
    *,
    group_col: str = "Stan wykończenia",
    value_col: str = "Cena_za_m2",
    label_map: dict | None = None,
    palette: str = "pastel",
    overlap: float = 0.5,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Rozkłady ceny za m² w zależności od stanu wykończenia z zaznaczoną medianą",
):
    """
    Joyplot (ridge plot) of price per m² grouped by finish state,
    with median marked for each group.
    """

    import joypy
    import seaborn as sns
    import matplotlib.pyplot as plt

    # --- walidacja
    for c in (group_col, value_col):
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' required by joyplot_price_by_finish_state")

    work = df[[group_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[group_col, value_col])

    # --- kolejność grup (alfabetyczna, stabilna)
    unique_labels = sorted(work[group_col].unique())

    # --- mapowanie nazw (do ładnych etykiet)
    if label_map is None:
        display_labels = unique_labels
    else:
        display_labels = [label_map.get(lbl, lbl) for lbl in unique_labels]

    # --- paleta kolorów
    colors = sns.color_palette(palette, n_colors=len(unique_labels))

    # --- joyplot
    fig, axes = joypy.joyplot(
        data=work,
        by=group_col,
        column=value_col,
        labels=display_labels,
        color=colors,
        overlap=overlap,
        figsize=figsize,
    )

    # --- mediany
    medians = work.groupby(group_col)[value_col].median()

    for ax, label in zip(axes, unique_labels):
        median_val = medians.loc[label]
        ax.axvline(
            median_val,
            color="black",
            linestyle="--",
            linewidth=2,
            zorder=10,
            ymin=0.13,
            ymax=0.7,
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.xlabel("Cena za m² (PLN)")
    plt.tight_layout()

    return fig, axes




@register_viz("amenities_frequency_bar_hue")
def amenities_frequency_bar_hue(
    df: pd.DataFrame,
    *,
    amenities_cols: list[str],
    min_frac: float = 0.05,
    title: str = "Liczba mieszkań z poszczególnymi udogodnieniami",
    xlabel: str = "Udogodnienie",
    ylabel: str = "Liczba wystąpień",
    figsize: tuple[int, int] = (12, 8),
    rotate: int = 45,
):
    """
    Barplot with ggplot-like 'scale_fill_hue()' pastel colors.
    Colors generated in HSL space for smooth hue transition.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import colorsys

    # --- walidacja
    if not amenities_cols:
        raise ValueError("amenities_cols must be provided in CFG")

    missing = [c for c in amenities_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing amenities columns: {missing}")

    # --- zliczanie wystąpień (jak w R: sum(.x, na.rm=TRUE))
    work = df[amenities_cols].apply(pd.to_numeric, errors="coerce")
    counts = work.fillna(0).astype(int).sum().sort_values(ascending=False)

    plot_df = counts.reset_index()
    plot_df.columns = ["Udogodnienie", "Liczba"]
    plot_df["Udogodnienie"] = plot_df["Udogodnienie"].str.replace(".", " ", regex=False)

    # --- próg 5%
    n = len(df)
    threshold = n * min_frac

    # --- generowanie kolorów HSL (jak ggplot2)
    def ggplot_hue(n, saturation=0.65, lightness=0.65):
        return [
            colorsys.hls_to_rgb(i / n, lightness, saturation)
            for i in range(n)
        ]

    colors = ggplot_hue(len(plot_df))

    # --- rysowanie
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        plot_df["Udogodnienie"],
        plot_df["Liczba"],
        color=colors
    )

    ax.axhline(threshold, linestyle="--", linewidth=1.8, color="black")

    ax.annotate(
        f"min. {int(min_frac*100)}% danych (n = {int(round(threshold))})",
        xy=(1.0, threshold),
        xycoords=("axes fraction", "data"),
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["Udogodnienie"], rotation=rotate, ha="right")

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    return fig, ax



@register_viz("distance_distributions_original_vs_log")
def distance_distributions_original_vs_log(
    df: pd.DataFrame,
    *,
    cols: list[str],
    label_map: dict[str, str] | None = None,
    xlim: tuple[float, float] | None = (0, 2500),
    alpha: float = 0.30,
    linewidth: float = 1.2,
    title_top: str = "Rozkłady oryginalne zmiennych",
    title_bottom: str = "Rozkłady logarytmiczne zmiennych",
    xlabel_top: str = "Odległość",
    xlabel_bottom: str = "Logarytmiczna odległość",
    ylabel: str = "Gęstość",
    legend_title: str = "zmienna",
    figsize: tuple[int, int] = (12, 6),
):
    """
    Two stacked density plots (original vs log1p) with one shared legend.
    Equivalent of ggplot + patchwork (p1 / p2) + guides='collect'.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- walidacja kolumn
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for distance_distributions_original_vs_log: {missing}")

    # --- przygotuj dane (bez inplace)
    work = df[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # --- etykiety jak w ggplot recode()
    def _label(c: str) -> str:
        if label_map and c in label_map:
            return label_map[c]
        return c

    # long format
    orig_long = work.melt(var_name="zmienna", value_name="wartosc").dropna()
    orig_long["zmienna"] = orig_long["zmienna"].map(_label)

    log_long = work.apply(np.log1p).melt(var_name="zmienna", value_name="wartosc").dropna()
    log_long["zmienna"] = log_long["zmienna"].map(_label)

    # --- plot
    plt.style.use("ggplot")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=False)
    ax1, ax2 = axes

    # spójna kolejność / kolory
    order = [_label(c) for c in cols]
    palette = sns.color_palette("husl", n_colors=len(order))
    pal_map = dict(zip(order, palette))

    # TOP: oryginalne
    for name in order:
        s = orig_long.loc[orig_long["zmienna"] == name, "wartosc"]
        if len(s) > 1:
            sns.kdeplot(
                x=s,
                ax=ax1,
                fill=True,
                alpha=alpha,
                linewidth=linewidth,
                color=pal_map[name],
                label=name,
            )

    ax1.set_title(title_top, loc="center")
    ax1.set_xlabel(xlabel_top)
    ax1.set_ylabel(ylabel)
    if xlim is not None:
        ax1.set_xlim(*xlim)

    # BOTTOM: logarytmiczne
    for name in order:
        s = log_long.loc[log_long["zmienna"] == name, "wartosc"]
        if len(s) > 1:
            sns.kdeplot(
                x=s,
                ax=ax2,
                fill=True,
                alpha=alpha,
                linewidth=linewidth,
                color=pal_map[name],
                label=name,
            )

    ax2.set_title(title_bottom, loc="center")
    ax2.set_xlabel(xlabel_bottom)
    ax2.set_ylabel(ylabel)

    # --- jedna legenda (zebrana)
    # zbieramy uchwyty tylko raz z ax1
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend_.remove() if ax1.get_legend() else None
    ax2.legend_.remove() if ax2.get_legend() else None

    fig.legend(
        handles,
        labels,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        frameon=False,
    )

    # miejsce na legendę po prawej
    fig.tight_layout(rect=[0, 0, 0.92, 1])

    return fig, axes




@register_viz("decision_tree_rooms_price")
def decision_tree_rooms_price(
    df: pd.DataFrame,
    *,
    feature_col: str = "Liczba pokoi",
    target_col: str = "Cena_za_m2",
    n_categories: int = 3,
    min_samples_leaf: int = 200,
    figsize: tuple[int, int] = (10, 4),
    title: str = "Drzewo decyzyjne: liczba pokoi → cena",
):
    """
    Drzewo decyzyjne dzielące dane na `n_categories` liści (kategorii),
    z kolorami liści gradientowo wg średniej ceny (jak rpart.plot RdYlGn).
    """

    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    import pandas as pd
    import numpy as np
    from matplotlib.cm import get_cmap
    from matplotlib.colors import Normalize

    # --- prepare data
    work = df[[feature_col, target_col]].copy()
    work[feature_col] = pd.to_numeric(work[feature_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna()

    X = work[[feature_col]]
    y = work[target_col]

    # --- model
    tree = DecisionTreeRegressor(
        max_leaf_nodes=n_categories,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    tree.fit(X, y)

    # --- plot
    fig, ax = plt.subplots(figsize=figsize)

    plot_tree(
        tree,
        feature_names=[feature_col],
        filled=False,          # ← WYŁĄCZAMY defaultowe kolory sklearn
        rounded=True,
        precision=0,
        ax=ax,
    )

    # --- przygotuj normalizację kolorów (RdYlGn)
    cmap = get_cmap("RdYlGn")
    norm = Normalize(vmin=y.min(), vmax=y.max())
    total_n = len(work)

    # --- popraw etykiety i kolory LIŚCI
    for t in ax.texts:
        txt = t.get_text()

        if "value =" not in txt:
            continue

        # --- średnia
        mean_val = float(
            txt.split("value =")[1]
               .split("\n")[0]
               .replace("[", "")
               .replace("]", "")
               .strip()
        )

        # --- liczność
        n = int(
            txt.split("samples =")[1]
               .split("\n")[0]
        )
        pct = int(round(100 * n / total_n))

        # --- kolor wg średniej
        color = cmap(norm(mean_val))

        # --- nowa etykieta (jak na screenie)
        t.set_text(
            f"{int(mean_val/1000)}e+3\n"
            f"n={n}  {pct}%"
        )

        # --- styl pudełka (jak rpart)
        t.set_bbox(dict(
            boxstyle="round,pad=0.35",
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
        ))

    ax.set_title(title)
    plt.tight_layout()
    return fig, ax



@register_viz("boxplot_price_by_district")
def boxplot_price_by_district(
    df: pd.DataFrame,
    *,
    district_col: str = "dzielnica",
    value_col: str = "cena_za_m2",
    title: str = "Rozkład ceny za m² w poszczególnych dzielnicach",
    figsize: tuple[int, int] = (12, 5),
    saturation: float = 0.65,
    lightness: float = 0.65,
):
    """
    Boxplot z gradientową paletą (ggplot-like hue sweep),
    dzielnice uporządkowane po medianie ceny.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import colorsys

    # --- dane
    work = df[[district_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()

    # --- reorder dzielnic po medianie (jak dplyr::reorder)
    medians = (
        work.groupby(district_col)[value_col]
            .median()
            .sort_values()
    )
    order = medians.index.tolist()

    # --- gradientowa paleta HSL (jak scale_fill_hue)
    def ggplot_hue(n, saturation=0.65, lightness=0.65):
        return [
            colorsys.hls_to_rgb(i / n, lightness, saturation)
            for i in range(n)
        ]

    colors = ggplot_hue(
        len(order),
        saturation=saturation,
        lightness=lightness,
    )

    palette = dict(zip(order, colors))

    # --- styl
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=work,
        x=district_col,
        y=value_col,
        order=order,
        palette=palette,
        ax=ax,
    )

    ax.set_title(title, loc="center")
    ax.set_xlabel("Dzielnica")
    ax.set_ylabel("Cena za m² (PLN)")
    ax.tick_params(axis="x", rotation=45)

    # --- brak legendy (jak guides(fill = "none"))
    ax.get_legend().remove() if ax.get_legend() else None

    plt.tight_layout()
    return fig, ax






@register_viz("mean_price_vs_distance_poly2")
def mean_price_vs_distance_poly2(
    df: pd.DataFrame,
    *,
    distance_cols: list[str] = (
        "dist_to_metro_m1",
        "dist_to_metro_m2",
        "dist_to_shop",
        "dist_to_green",
    ),
    price_col: str = "cena_za_m2",
    bins: int = 100,
    poly_degree: int = 2,
    figsize: tuple[int, int] = (12, 8),
    title: str = "Średnia cena za m² w zależności od odległości z trendem wielomianowym",
):
    """
    Pythonowy odpowiednik:
    - cut(..., breaks = 100)
    - mean(price) w binach
    - geom_line + geom_point
    - geom_smooth(method = 'lm', formula = y ~ poly(x, 2))
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    import seaborn as sns

    # --- walidacja
    missing = [c for c in distance_cols + [price_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # --- long format
    long = (
        df[distance_cols + [price_col]]
        .dropna()
        .melt(
            id_vars=price_col,
            value_vars=distance_cols,
            var_name="distance_type",
            value_name="distance",
        )
    )

    # --- binning (cut)
    long["bin"] = (
        long
        .groupby("distance_type")["distance"]
        .transform(lambda x: pd.cut(x, bins=bins, labels=False))
    )

    # --- agregacja
    grouped = (
        long
        .groupby(["distance_type", "bin"], as_index=False)
        .agg(
            mean_price=(price_col, "mean"),
            min_dist=("distance", "min"),
            max_dist=("distance", "max"),
        )
    )

    grouped["distance_mid"] = (grouped["min_dist"] + grouped["max_dist"]) / 2

    # --- etykiety (jak w R)
    facet_titles = {
        "dist_to_metro_m1": "Odległość od stacji M1",
        "dist_to_metro_m2": "Odległość od stacji M2",
        "dist_to_shop": "Odległość od sklepu",
        "dist_to_green": "Odległość od zieleni",
    }

    # --- styl
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2", n_colors=len(distance_cols))

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=False)
    axes = axes.flatten()

    for ax, dist_col, color in zip(axes, distance_cols, palette):
        sub = grouped[grouped["distance_type"] == dist_col]

        # linia + punkty
        ax.plot(
            sub["distance_mid"],
            sub["mean_price"],
            color=color,
            linewidth=1.2,
        )
        ax.scatter(
            sub["distance_mid"],
            sub["mean_price"],
            color=color,
            s=25,
        )

        # --- trend kwadratowy
        X = sub[["distance_mid"]].values
        y = sub["mean_price"].values

        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        x_fit = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        y_fit = model.predict(poly.transform(x_fit))

        ax.plot(x_fit, y_fit, color="black", linewidth=2)

        ax.set_title(facet_titles.get(dist_col, dist_col))
        ax.set_xlabel("Środek przedziału odległości (m)")
        ax.set_ylabel("Średnia cena za m²")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, axes



@register_viz("warsaw_wordcloud")
def warsaw_wordcloud(
    df: pd.DataFrame,
    *,
    text_col: str,                 # ← podasz w CFG
    geojson_path: str = "geojson/warsaw_districts.geojson",
    max_words: int = 80,
    min_word_length: int = 3,
    background_color: str = "white",
    colormap: str = "tab10",
    title: str = "Najczęściej występujące pojęcia w opisach mieszkań",
    figsize: tuple[int, int] = (8, 8),
):
    """
    Wordcloud ograniczony do obszaru Warszawy,
    z automatycznymi stopwords (PL + EN),
    top-N najczęstszych słów,
    czyszczenie znaków specjalnych.
    """

    import re
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    from wordcloud import WordCloud, STOPWORDS
    from stopwordsiso import stopwords
    from shapely.ops import unary_union

    # --- walidacja
    if text_col not in df.columns:
        raise KeyError(f"Missing column '{text_col}' required by warsaw_wordcloud")

    # ============================================================
    # STOPWORDS (AUTOMATYCZNE)
    # ============================================================
    STOPWORDS_PL = set(stopwords("pl"))
    ALL_STOPWORDS = STOPWORDS.union(STOPWORDS_PL)

    # ============================================================
    # TEKST → TOKENY
    # ============================================================
    text = (
        df[text_col]
        .dropna()
        .astype(str)
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)   # znaki specjalne
        .str.replace(r"\d+", " ", regex=True)       # liczby
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.cat(sep=" ")
    )

    tokens = [
        t for t in text.split()
        if len(t) >= min_word_length
        and t not in ALL_STOPWORDS
    ]

    if not tokens:
        raise ValueError("No valid tokens after stopword filtering")

    clean_text = " ".join(tokens)

    # ============================================================
    # MASKA WARSZAWY
    # ============================================================
    warsaw = gpd.read_file(geojson_path)
    if warsaw.crs is None:
        warsaw = warsaw.set_crs("EPSG:4326")

    geom = unary_union(warsaw.geometry)
    minx, miny, maxx, maxy = geom.bounds

    width = 800
    height = 800

    xs = np.linspace(minx, maxx, width)
    ys = np.linspace(maxy, miny, height)
    xx, yy = np.meshgrid(xs, ys)


    mask = np.array([
        geom.contains(gpd.points_from_xy([x], [y])[0])
        for x, y in zip(xx.ravel(), yy.ravel())
    ]).reshape((height, width))

    mask = np.where(mask, 0, 255).astype(np.uint8)


    # ============================================================
    # WORDCLOUD
    # ============================================================
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        stopwords=ALL_STOPWORDS,
        colormap=colormap,
        mask=mask,
        contour_width=1.2,
        contour_color="black",
    ).generate(clean_text)

    # ============================================================
    # PLOT
    # ============================================================
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig, ax
