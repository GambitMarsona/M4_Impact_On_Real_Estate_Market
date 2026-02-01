"""
machine_learning_and_visualisations.py

Ten moduł korzysta z konfiguracji CFG_ML do zbudowania spójnego pipeline’u:
- przygotowania danych (feature engineering + preprocessing),
- trenowania i walidacji modeli,
- oraz wygenerowania zestawu wizualizacji diagnostycznych i opisowych.

Konfiguracja (CFG_ML) — kluczowe sekcje:

1) CFG_ML["experiment"]
   - definicja eksperymentu ML:
     - target_col: kolumna celu (np. "Cena_za_m2")
     - log_target: czy zastosować logarytmowanie targetu
     - test_size: rozmiar zbioru testowego
     - random_state: ziarno losowości (powtarzalność wyników)
     - cv_folds: liczba foldów w walidacji krzyżowej
     - scoring: metryki oceny (MAE, RMSE, R2)

2) CFG_ML["features"]
   - podział zmiennych według typu statystycznego (do poprawnego preprocessingu):
     - continuous: zmienne ciągłe (np. odległości, metraż)
     - ordinal: zmienne porządkowe (np. piętro, rok budowy)
     - categorical: zmienne nominalne do kodowania (np. dzielnica, stan wykończenia)
     - binary: zmienne 0/1 (udogodnienia)

3) CFG_ML["preprocessing"]
   - transformacje wejściowe wykonywane przed modelowaniem:
     - scaling: metoda skalowania (np. standard) i zakres zastosowania
     - encoding: kodowanie kategorii (one-hot) + drop_first
     - log_transform: lista kolumn, na których wykonywany jest log-transform
       (np. odległości do metra / terenów zielonych / supermarketu)

4) CFG_ML["models"]
   - lista modeli do uruchomienia wraz z siatkami hiperparametrów:
     - random_forest
     - xgboost
     - svr
     - dnn
   - każdy wpis zawiera:
     - name: identyfikator modelu
     - params: zakres hiperparametrów do strojenia (GridSearch / podobne)

5) CFG_ML["training"]
   - ustawienia trenowania dla modelu DNN:
     - optimizer, epochs, batch_size, validation_split
     - early_stopping (patience, restore_best_weights)

Wizualizacje / raporty generowane przez moduł (rejestrowane i uruchamiane przez runner):
  - "ml_true_vs_pred_kde_grid"
  - "ml_feature_importance_all_models"
  - "ml_feature_importance_rf_vs_xgboost"
  - "ml_metrics_bar_models"
  - "ml_poly_degree_mse_panel"
  - "ml_distance_effect_poly10_panel"
  - "_ml_gradient_map_max_envelope"
  
Dodatkowo:
- generowany jest raport brakujących danych przed uruchomieniem wizualizacji,
- wyniki wizualizacji są automatycznie zapisywane do katalogu ./visualizations
  (katalog jest tworzony, jeżeli nie istnieje).
"""


from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import numpy as np
import pandas as pd
from core import Bundle, Pipeline, StepExecutionError
from visualizations_and_filtering import register_viz


# ============================================================
# Helpers
# ============================================================

def _sanitize_col(name: str) -> str:
    s = str(name).strip().lower()
    translit_map = {
        "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n",
        "ó": "o", "ś": "s", "ż": "z", "ź": "z",
    }
    for src, tgt in translit_map.items():
        s = s.replace(src, tgt)
    s = s.replace(" ", "_")
    s = "".join(ch for ch in s if (ch.isalnum() or ch == "_"))
    s = s.strip("_")
    return s or "unknown"


def _non_null_count(df: pd.DataFrame, col: str) -> int:
    try:
        return int(df[col].notna().sum())
    except Exception:
        return 0


def _resolve_columns(
    df: pd.DataFrame,
    cols: Iterable[str],
    ) -> Tuple[List[str], Dict[str, str], List[str]]:
    resolved: List[str] = []
    mapping: Dict[str, str] = {}
    missing: List[str] = []

    df_cols = set(df.columns)

    for c in cols:
        exact = c if c in df_cols else None
        sc = _sanitize_col(c)
        sanitized = sc if sc in df_cols else None

        chosen = None
        if exact and sanitized and exact != sanitized:
            chosen = sanitized if _non_null_count(df, sanitized) > _non_null_count(df, exact) else exact
        elif exact:
            chosen = exact
        elif sanitized:
            chosen = sanitized

        if chosen is None:
            missing.append(c)
        else:
            resolved.append(chosen)
            mapping[c] = chosen

    return resolved, mapping, missing


def _as_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _save_fig(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    return str(path)


def _in_notebook() -> bool:
    """Detekcja czy kod jest odpalany w Jupyter Notebooku"""
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is None:
            return False
        # działa na jupyter + VS CODE
        return bool(getattr(ip, "kernel", None)) or bool(ip.config.get("IPKernelApp"))
    except Exception:
        return False


def _maybe_show_matplotlib(fig, *, enabled: bool) -> None:
    if not enabled:
        return
    try:
        import matplotlib.pyplot as plt
        try:
            plt.figure(fig.number)  
        except Exception:
            pass
        plt.show()
    except Exception:
        pass


def _metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "mse": mse, "r2": r2}


def _safe_get_feature_names(preprocessor: Any) -> Optional[np.ndarray]:
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return None


def _build_preprocessor(
    df: pd.DataFrame,
    *,
    CFG: Mapping[str, Any],
) -> Tuple[Any, Dict[str, List[str]]]:
    """
    Preprocessing zgodny z CFG_ML:
    - numeric: imputer median + (opcjonalnie) log1p + (opcjonalnie) scaling
    - categorical: imputer most_frequent + onehot
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler

    feats = CFG.get("features", {}) or {}
    pre = CFG.get("preprocessing", {}) or {}
    scaling = (pre.get("scaling") or {})
    encoding = (pre.get("encoding") or {})

    continuous_raw = list(feats.get("continuous", []) or [])
    ordinal_raw = list(feats.get("ordinal", []) or [])
    categorical_raw = list(feats.get("categorical", []) or [])
    binary_raw = list(feats.get("binary", []) or [])

    continuous, _, miss_c = _resolve_columns(df, continuous_raw)
    ordinal, _, miss_o = _resolve_columns(df, ordinal_raw)
    categorical, _, miss_cat = _resolve_columns(df, categorical_raw)
    binary, _, miss_b = _resolve_columns(df, binary_raw)

    missing = miss_c + miss_o + miss_cat + miss_b
    if missing:
        raise StepExecutionError(
            "ML preprocessing: brakuje kolumn z CFG['features']: "
            f"{missing}. Dostępne kolumny (przykład): {list(df.columns)[:40]}"
        )

    log_cols_raw = list(pre.get("log_transform", []) or [])
    log_cols, _, miss_log = _resolve_columns(df, log_cols_raw)
    if miss_log:
        raise StepExecutionError(
            "ML preprocessing: brakuje kolumn z CFG['preprocessing']['log_transform']: "
            f"{miss_log}"
        )

    method = (scaling.get("method") or "standard").lower()
    apply_to = (scaling.get("apply_to") or "continuous").lower()

    def _mk_scaler():
        if method == "none":
            return None
        if method == "minmax":
            return MinMaxScaler()
        if method == "standard":
            return StandardScaler()
        raise StepExecutionError("ML preprocessing: scaling['method'] musi być: standard|minmax|none")

    scaler = _mk_scaler()

    def _mk_num_pipe(*, with_log: bool, with_scale: bool) -> SkPipeline:
        steps: List[Tuple[str, Any]] = []
        steps.append(("imputer", SimpleImputer(strategy="median")))
        if with_log:
            steps.append(
                ("log", FunctionTransformer(lambda x: np.log1p(np.maximum(x, 0)), feature_names_out="one-to-one"))
            )
        if with_scale and scaler is not None:
            steps.append(("scaler", scaler))
        return SkPipeline(steps)

    def _mk_other_num_pipe(*, with_scale: bool) -> SkPipeline:
        steps: List[Tuple[str, Any]] = []
        steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        if with_scale and scaler is not None:
            steps.append(("scaler", scaler))
        return SkPipeline(steps)

    drop_first = bool(encoding.get("drop_first", True))
    cat_mode = (encoding.get("categorical") or "onehot").lower()
    if cat_mode != "onehot":
        raise StepExecutionError("ML preprocessing: wspieram tutaj tylko encoding['categorical'] == 'onehot'.")

    # kompatybilność sklearn
    try:
        ohe = OneHotEncoder(drop="first" if drop_first else None, handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(drop="first" if drop_first else None, handle_unknown="ignore", sparse=False)

    cat_imputer = SimpleImputer(strategy="most_frequent")

    # ciągłe: rozdzielamy na log i non-log
    cont_log = [c for c in continuous if c in log_cols]
    cont_plain = [c for c in continuous if c not in log_cols]

    scale_cont = apply_to in ("continuous", "all")
    scale_other = apply_to == "all"

    transformers: List[Tuple[str, Any, List[str]]] = []
    if cont_log:
        transformers.append(("cont_log", _mk_num_pipe(with_log=True, with_scale=scale_cont), cont_log))
    if cont_plain:
        transformers.append(("cont", _mk_num_pipe(with_log=False, with_scale=scale_cont), cont_plain))
    if ordinal:
        transformers.append(("ord", _mk_other_num_pipe(with_scale=scale_other), ordinal))
    if binary:
        transformers.append(("bin", _mk_other_num_pipe(with_scale=scale_other), binary))
    if categorical:
        transformers.append(("cat", SkPipeline([("imputer", cat_imputer), ("ohe", ohe)]), categorical))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    colsets = {
        "continuous": continuous,
        "ordinal": ordinal,
        "binary": binary,
        "categorical": categorical,
        "log_transform": log_cols,
    }
    return ct, colsets


def _human_model_name(internal_name: str) -> str:
    m = str(internal_name).strip().lower()
    return {
        "svr": "SVM",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "dnn": "Deep NN",
    }.get(m, internal_name)


def _feature_rename(feature: str, rename_dict: Mapping[str, str]) -> str:
    key = _sanitize_col(feature)
    if key in rename_dict:
        return str(rename_dict[key])
    return feature


def _make_default_feature_rename_dict() -> Dict[str, str]:
    return {
        "winda": "Winda",
        "garaz": "Garaż",
        "internet": "Internet",
        "klimatyzacja": "Klimatyzacja",
        "lodowka": "Lodówka",
        "meble": "Meble",
        "ochrona": "Ochrona",
        "balkon": "Balkon",
        "oddzielna_kuchnia": "Oddzielna kuchnia",
        "ogrodek": "Ogródek",
        "piwnica": "Piwnica",
        "pom_uzytkowe": "Pomieszczenia użytkowe",
        "pralka": "Pralka",
        "system_alarmowy": "System alarmowy",
        "taras": "Taras",
        "telefon": "Telefon",
        "telewizor": "Telewizor",
        "teren_zamkniety": "Teren zamknięty",
        "zmywarka": "Zmywarka",
        "log_dist_to_metro_m1": "Odległość do metra M1",
        "log_dist_to_metro_m2": "Odległość do metra M2",
        "log_dist_to_green": "Odległość do zieleni",
        "log_dist_to_shop": "Odległość do sklepu",
        "dist_metro_m1_m": "Odległość do metra M1",
        "dist_metro_m2_m": "Odległość do metra M2",
        "dist_green_m": "Odległość do zieleni",
        "dist_supermarket_m": "Odległość do sklepu",
    }


def _permutation_importance_any_model(
    *,
    model_name: str,
    model_obj: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names_out: Optional[np.ndarray],
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:

    from sklearn.inspection import permutation_importance

    m = str(model_name).lower()

    # SKLEARN PIPELINE / ESTYMATOR
    if m != "dnn":
        res = permutation_importance(
            model_obj,
            X_test,
            y_test,
            scoring="neg_mean_squared_error",
            n_repeats=int(n_repeats),
            random_state=int(random_state),
            n_jobs=-1,
        )

        if isinstance(X_test, pd.DataFrame):
            feats = np.array(list(X_test.columns), dtype=object)
        else:
            feats = np.array([f"f{i}" for i in range(res.importances_mean.shape[0])], dtype=object)

        k = min(len(feats), int(res.importances_mean.shape[0]))
        return pd.DataFrame({"feature": feats[:k], "importance": res.importances_mean[:k]})

    # DNN 
    try:
        pre = model_obj["preprocessor"]
        keras_model = model_obj["model"]
    except Exception as e:
        raise StepExecutionError("Permutation importance: DNN artifacts mają złą strukturę.") from e

    Xte = np.asarray(pre.transform(X_test), dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    rng = np.random.default_rng(int(random_state))
    base = keras_model.predict(Xte, verbose=0).reshape(-1)
    base_mse = float(np.mean((y_test - base) ** 2))

    importances = np.zeros(Xte.shape[1], dtype=float)
    for j in range(Xte.shape[1]):
        mses = []
        for _ in range(int(n_repeats)):
            Xp = Xte.copy()
            col = Xp[:, j].copy()
            rng.shuffle(col)
            Xp[:, j] = col
            pred = keras_model.predict(Xp, verbose=0).reshape(-1)
            mses.append(float(np.mean((y_test - pred) ** 2)))
        importances[j] = float(np.mean(mses) - base_mse)

    feats = feature_names_out
    if feats is None or len(feats) != Xte.shape[1]:
        feats = np.array([f"f{i}" for i in range(Xte.shape[1])], dtype=object)

    return pd.DataFrame({"feature": feats, "importance": importances})


# ============================================================
# Wizualizacja: prawdziwe wartości vs przewidziane wartości
# ============================================================
@register_viz("ml_true_vs_pred_kde_grid")
def ml_true_vs_pred_kde_grid(
    df: pd.DataFrame,
    *,
    true_col: str = "y_true",
    ncols: int = 2,
    figsize: tuple[int, int] = (16, 12),
    cmap: str = "viridis",
    s: int = 10,
    title_prefix: str = "",
    xlabel: str = "Rzeczywiste ceny za m²",
    ylabel: str = "Przewidywane ceny za m²",
):
    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    if true_col not in df.columns:
        raise KeyError(f"Missing column '{true_col}'")

    pred_cols = [c for c in df.columns if c.startswith("y_pred_")]
    if not pred_cols:
        raise KeyError("No prediction columns found. Expected columns like 'y_pred_*'.")

    models = [c.replace("y_pred_", "") for c in pred_cols]
    y_true_all = pd.to_numeric(df[true_col], errors="coerce").to_numpy()

    n = len(models)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], max(figsize[1], 6 * nrows)))
    axes = np.array(axes).reshape(-1)

    for i, model_name in enumerate(models):
        y_pred_all = pd.to_numeric(df[f"y_pred_{model_name}"], errors="coerce").to_numpy()

        ok = ~np.isnan(y_true_all) & ~np.isnan(y_pred_all)
        y_true = y_true_all[ok]
        y_pred = y_pred_all[ok]

        ax = axes[i]
        if y_true.size < 5:
            ax.set_title(f"{title_prefix}{_human_model_name(model_name)}".strip())
            ax.text(0.5, 0.5, "Za mało danych", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        xy = np.vstack([y_true, y_pred])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x_sorted, y_sorted, z_sorted = y_true[idx], y_pred[idx], z[idx]

        reg = LinearRegression().fit(x_sorted.reshape(-1, 1), y_sorted)
        y_fit = reg.predict(x_sorted.reshape(-1, 1))

        ax.scatter(x_sorted, y_sorted, c=z_sorted, cmap=cmap, s=s, label="Wartości")
        ax.plot(x_sorted, y_fit, color="red", label="Linia regresji")

        mn = float(min(x_sorted.min(), y_sorted.min()))
        mx = float(max(x_sorted.max(), y_sorted.max()))
        ax.plot([mn, mx], [mn, mx], color="green", linestyle="--", label="Idealna predykcja")

        ax.set_title(f"{title_prefix}{_human_model_name(model_name)}".strip(), fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.supxlabel(xlabel, fontsize=25)
    fig.supylabel(ylabel, fontsize=25)
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.96])
    return fig, axes

# ============================================================
# Wizualizacja: Porównanie ważności cech: wszystkie modele
# ============================================================
@register_viz("ml_feature_importance_all_models")
def ml_feature_importance_all_models(
    df_importances: pd.DataFrame,
    *,
    top_n: int = 20,
    title: str = "Porównanie ważności cech: wszystkie modele",
    figsize: tuple[int, int] = (12, 9),
):
    import matplotlib.pyplot as plt

    d = df_importances.copy()
    d["importance"] = pd.to_numeric(d["importance"], errors="coerce")
    d = d.dropna(subset=["importance", "feature", "model"])

    agg = (
        d.groupby("feature", as_index=False)["importance"]
        .mean()
    )
    agg = agg.sort_values("importance", ascending=False).head(int(top_n))
    top_feats = list(agg["feature"])
    d = d[d["feature"].isin(top_feats)].copy()

    order = top_feats
    d["feature"] = pd.Categorical(d["feature"], categories=order, ordered=True)

    models = list(d["model"].unique())
    m = len(models)

    piv = d.pivot_table(index="feature", columns="model", values="importance", aggfunc="mean").fillna(0.0)
    piv = piv.loc[order]

    y = np.arange(len(order))
    bar_width = 0.8 / max(1, m)

    fig, ax = plt.subplots(figsize=figsize)

    color_map = {
        "Random Forest": "#00bfff",
        "XGBoost": "#ff1493",
        "SVM": "#7cfc00",
        "Deep NN": "#ffa500",
    }

    for i, model in enumerate(models):
        vals = piv[model].values if model in piv.columns else np.zeros_like(y, dtype=float)
        ax.barh(
            y + (i - (m - 1) / 2) * bar_width,
            vals,
            height=bar_width,
            label=model,
            color=color_map.get(model, None),
        )

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Ważność cechy", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig, ax


# ============================================================
# Wizualizacja: Porównanie ważności cech: Random Forest vs XGBoos
# ============================================================
@register_viz("ml_feature_importance_rf_vs_xgboost")
def ml_feature_importance_rf_vs_xgboost(
    df_importances: pd.DataFrame,
    *,
    top_n: int = 20,
    title: str = "Porównanie ważności cech: Random Forest vs XGBoost",
    figsize: tuple[int, int] = (10, 8),
    rf_label: str = "Random Forest",
    xgb_label: str = "XGBoost",
):
    import matplotlib.pyplot as plt

    d = df_importances.copy()
    need = {"feature", "importance", "model"}
    if not need.issubset(d.columns):
        raise KeyError(f"df_importances musi mieć kolumny: {sorted(need)}")

    d["importance"] = pd.to_numeric(d["importance"], errors="coerce")
    d["feature"] = d["feature"].astype(str)
    d["model"] = d["model"].astype(str)
    d = d.dropna(subset=["importance", "feature", "model"]).copy()

    d = d[d["model"].isin([rf_label, xgb_label])].copy()
    if d.empty:
        raise StepExecutionError(
            f"Brak w df_importances modeli '{rf_label}' i/lub '{xgb_label}'. "
            f"Dostępne: {sorted(df_importances['model'].dropna().unique().tolist())}"
        )

    # wybieramy top cechy po średniej ważności 
    piv = (
        d.pivot_table(index="feature", columns="model", values="importance", aggfunc="mean")
        .fillna(0.0)
    )
    # upewnienie się, że obie kolumny istnieją
    if rf_label not in piv.columns:
        piv[rf_label] = 0.0
    if xgb_label not in piv.columns:
        piv[xgb_label] = 0.0

    piv["_mean"] = (piv[rf_label] + piv[xgb_label]) / 2.0
    piv = piv.sort_values("_mean", ascending=False).head(int(top_n))
    piv = piv.drop(columns=["_mean"])

    feats = piv.index.tolist()
    rf_vals = piv[rf_label].to_numpy(dtype=float)
    xgb_vals = piv[xgb_label].to_numpy(dtype=float)
    y = np.arange(len(feats))

    bar_width = 0.4
    fig, ax = plt.subplots(figsize=figsize)

    rf_color = "#00bfff"  
    xgb_color = "#ff1493"  

    ax.barh(y - bar_width / 2, rf_vals, bar_width, label=rf_label, color=rf_color)
    ax.barh(y + bar_width / 2, xgb_vals, bar_width, label=xgb_label, color=xgb_color)

    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Ważność cechy", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig, ax

# ============================================================
# Wizualizacja: Porównanie jakości modeli na zbiorze testowym
# ============================================================
@register_viz("ml_metrics_bar_models")
def ml_metrics_bar_models(
    df_metrics: pd.DataFrame,
    *,
    figsize: tuple[int, int] = (14, 6),
    title: str = "Porównanie jakości modeli na zbiorze testowym",
):

    import matplotlib.pyplot as plt

    d = df_metrics.copy()

    # walidacja
    if "model" not in d.columns:
        raise KeyError("df_metrics musi mieć kolumnę: 'model'")
    if "mae" not in d.columns:
        raise KeyError("df_metrics musi mieć kolumnę: 'mae'")
    if "r2" not in d.columns:
        raise KeyError("df_metrics musi mieć kolumnę: 'r2'")
    metric_mid = "mse" if "mse" in d.columns else ("rmse" if "rmse" in d.columns else None)
    if metric_mid is None:
        raise KeyError("df_metrics musi mieć kolumnę: 'mse' albo 'rmse'")

    # sortowanie
    d[["mae", "r2", metric_mid]] = d[["mae", "r2", metric_mid]].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=["mae", "r2", metric_mid])
    d = d.sort_values(metric_mid, ascending=True).reset_index(drop=True)

    labels = d["model"].astype(str).tolist()
    n = len(labels)

    # paleta 
    try:
        import seaborn as sns  
        palette = sns.color_palette("Set2", n_colors=max(1, n))
        colors = [palette[i] for i in range(n)]
    except Exception:
        cmap = plt.get_cmap("Set2")
        colors = [cmap(i / max(1, n - 1)) for i in range(n)]

    color_map = {lab: colors[i] for i, lab in enumerate(labels)}
    bar_colors = [color_map[l] for l in labels]

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # MAE
    axs[0].bar(labels, d["mae"].values, color=bar_colors)
    axs[0].set_title("MAE")
    axs[0].tick_params(axis="x", rotation=15)

    # MSE / RMSE
    mid_title = "MSE" if metric_mid == "mse" else "RMSE"
    axs[1].bar(labels, d[metric_mid].values, color=bar_colors)
    axs[1].set_title(mid_title)
    axs[1].tick_params(axis="x", rotation=15)

    # R2
    axs[2].bar(labels, d["r2"].values, color=bar_colors)
    axs[2].set_title("R²")
    axs[2].tick_params(axis="x", rotation=15)

    for ax in axs:
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_ylabel("Wartość")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig, axs

# ============================================================
# Wizualizacja: Porównanie wartości MSE dla różnych modeli
# ============================================================
@register_viz("ml_poly_degree_mse_panel")
def ml_poly_degree_mse_panel(
    df_stats: pd.DataFrame,
    *,
    figsize: tuple[int, int] = (14, 10),
    title: str = "Porównanie wartości MSE dla różnych modeli",
):
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except Exception as e:
        raise StepExecutionError("Brak seaborn. Zainstaluj: pip install seaborn") from e

    required = {"stopień", "MSE_M2_XGBoost", "MSE_M2_Rf", "MSE_M1_XGBoost", "MSE_M1_Rf"}
    if not required.issubset(df_stats.columns):
        raise KeyError(f"df_stats musi mieć kolumny: {sorted(required)}")

    sns.set_context("talk")
    sns.set(style="whitegrid", palette="pastel")

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    sns.lineplot(ax=axs[0, 0], x="stopień", y="MSE_M2_XGBoost", data=df_stats, color="red", marker="o", linewidth=4)
    axs[0, 0].set_title("M2 XGBoost", fontsize=16)
    axs[0, 0].set_xlabel("Stopień Wielomianu", fontsize=14)
    axs[0, 0].set_ylabel("MSE", fontsize=14)

    sns.lineplot(ax=axs[0, 1], x="stopień", y="MSE_M2_Rf", data=df_stats, color="red", marker="o", linewidth=4)
    axs[0, 1].set_title("M2 RF", fontsize=16)
    axs[0, 1].set_xlabel("Stopień Wielomianu", fontsize=14)
    axs[0, 1].set_ylabel("MSE", fontsize=14)

    sns.lineplot(ax=axs[1, 0], x="stopień", y="MSE_M1_XGBoost", data=df_stats, color="blue", marker="o", linewidth=4)
    axs[1, 0].set_title("M1 XGBoost", fontsize=16)
    axs[1, 0].set_xlabel("Stopień Wielomianu", fontsize=14)
    axs[1, 0].set_ylabel("MSE", fontsize=14)

    sns.lineplot(ax=axs[1, 1], x="stopień", y="MSE_M1_Rf", data=df_stats, color="blue", marker="o", linewidth=4)
    axs[1, 1].set_title("M1 RF", fontsize=16)
    axs[1, 1].set_xlabel("Stopień Wielomianu", fontsize=14)
    axs[1, 1].set_ylabel("MSE", fontsize=14)

    for ax in axs.flat:
        sns.despine(ax=ax, top=True, right=True)
        ax.tick_params(labelsize=12)

    plt.suptitle(title, fontsize=18, y=1.02)
    plt.tight_layout()
    return fig, axs

# ============================================================
# Wizualizacja: Porównanie wpływu odległości od Metra na cenę za m²
# ============================================================
@register_viz("ml_distance_effect_poly10_panel")
def ml_distance_effect_poly10_panel(
    fikcyjne: pd.DataFrame,
    *,
    figsize: tuple[int, int] = (15, 12),
    title: str = "Porównanie wpływu odległości od Metra na cenę za m²",
):
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except Exception as e:
        raise StepExecutionError("Brak seaborn. Zainstaluj: pip install seaborn") from e
    from scipy.optimize import curve_fit

    required = {
        "log_dist_to_metro_m1",
        "log_dist_to_metro_m2",
        "cena_predicted_m1_rf",
        "cena_predicted_m2_rf",
        "cena_predicted_m1_xgboost",
        "cena_predicted_m2_xgboost",
    }
    if not required.issubset(fikcyjne.columns):
        raise KeyError(f"fikcyjne musi mieć kolumny: {sorted(required)}")

    sns.set(style="whitegrid")

    def poly10(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
        return (
            a0 * x**10 + a1 * x**9 + a2 * x**8 + a3 * x**7 + a4 * x**6 +
            a5 * x**5 + a6 * x**4 + a7 * x**3 + a8 * x**2 + a9 * x + a10
        )

    x_m1 = np.expm1(pd.to_numeric(fikcyjne["log_dist_to_metro_m1"], errors="coerce").to_numpy())
    x_m2 = np.expm1(pd.to_numeric(fikcyjne["log_dist_to_metro_m2"], errors="coerce").to_numpy())

    y_m1_rf  = pd.to_numeric(fikcyjne["cena_predicted_m1_rf"], errors="coerce").to_numpy()
    y_m2_rf  = pd.to_numeric(fikcyjne["cena_predicted_m2_rf"], errors="coerce").to_numpy()
    y_m1_xgb = pd.to_numeric(fikcyjne["cena_predicted_m1_xgboost"], errors="coerce").to_numpy()
    y_m2_xgb = pd.to_numeric(fikcyjne["cena_predicted_m2_xgboost"], errors="coerce").to_numpy()

    # globalne granice
    x_all = np.concatenate([x_m1[~np.isnan(x_m1)], x_m2[~np.isnan(x_m2)]]) if (np.any(~np.isnan(x_m1)) and np.any(~np.isnan(x_m2))) else None
    y_all = np.concatenate([
        y_m1_rf[~np.isnan(y_m1_rf)], y_m2_rf[~np.isnan(y_m2_rf)],
        y_m1_xgb[~np.isnan(y_m1_xgb)], y_m2_xgb[~np.isnan(y_m2_xgb)]
    ])
    global_x_min = float(np.nanmin(x_all)) if x_all is not None and len(x_all) else 0.0
    global_x_max = float(np.nanmax(x_all)) if x_all is not None and len(x_all) else 1.0
    global_y_min = float(np.nanmin(y_all)) if len(y_all) else 0.0
    global_y_max = float(np.nanmax(y_all)) if len(y_all) else 1.0

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    def _panel(ax, x, y, color, panel_title):
        ok = ~np.isnan(x) & ~np.isnan(y)
        x1 = x[ok]
        y1 = y[ok]
        sns.scatterplot(x=x1, y=y1, color=color, ax=ax)
        ax.set_title(panel_title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if len(x1) >= 20:
            params, _ = curve_fit(poly10, x1, y1, maxfev=200000)
            x_fit = np.linspace(float(np.min(x1)), float(np.max(x1)), 1000)
            y_fit = poly10(x_fit, *params)
            ax.plot(x_fit, y_fit, color="black", linewidth=4, label="Funkcja interpolująca")
            ax.legend()

        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)

    _panel(axs[0, 0], x_m2, y_m2_xgb, "red", "M2 – XGBoost")
    _panel(axs[0, 1], x_m2, y_m2_rf,  "red", "M2 – RF")
    _panel(axs[1, 0], x_m1, y_m1_xgb, "blue", "M1 – XGBoost")
    _panel(axs[1, 1], x_m1, y_m1_rf,  "blue", "M1 – RF")

    fig.text(0.5, 0.04, "Odległość od metra (m)", ha="center", va="center", fontsize=25)
    fig.text(0.03, 0.5, "Cena przewidziana (PLN)", ha="center", va="center", rotation="vertical", fontsize=25)
    plt.suptitle(title, fontsize=25, y=0.93)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    return fig, axs

def _make_typical_row_from_train(X_ref: pd.DataFrame) -> Dict[str, Any]:
    """
    Buduje "najbardziej przeciętne mieszkanie" z danych TRAIN (X_ref):
    - numeric -> median
    - non-numeric -> mode
    Zwraca dict: kolumna -> wartość.
    """
    base: Dict[str, Any] = {}
    for c in X_ref.columns:
        s = X_ref[c]
        if pd.api.types.is_numeric_dtype(s):
            base[c] = float(pd.to_numeric(s, errors="coerce").median())
        else:
            base[c] = str(s.mode(dropna=True).iloc[0]) if s.notna().any() else ""
    return base


def _predict_curve_from_model(
    *,
    model_obj: Any,
    X_ref: pd.DataFrame,
    distance_col: str,
    r_grid_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robi krzywą f(r) bez wielomianu:
      f(r) = model.predict(typowe_mieszkanie, distance_col=r)
    Zwraca (r_grid_m, y_hat).
    """
    base = _make_typical_row_from_train(X_ref)
    rows = []
    for r in r_grid_m:
        rr = dict(base)
        rr[distance_col] = float(r)
        rows.append(rr)
    df_syn = pd.DataFrame(rows, columns=X_ref.columns)  # pilnuj kolejności/kolumn
    y_hat = np.asarray(model_obj.predict(df_syn), dtype=float).reshape(-1)
    return np.asarray(r_grid_m, dtype=float), y_hat


def _interp_f(r_grid: np.ndarray, y_grid: np.ndarray):
    """
    Zwraca funkcję f(r) działającą na numpy array (w metrach).
    - dla r < min -> wartość na min
    - dla r > max -> wartość na max (dalej i tak zwykle ucinamy max_distance)
    """
    r_grid = np.asarray(r_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)

    # usuń NaN i posortuj po r
    ok = ~np.isnan(r_grid) & ~np.isnan(y_grid)
    r = r_grid[ok]
    y = y_grid[ok]
    if r.size < 2:
        raise StepExecutionError("Interpolacja f(r): za mało punktów do zbudowania funkcji.")
    idx = np.argsort(r)
    r = r[idx]
    y = y[idx]

    def f(rq: np.ndarray) -> np.ndarray:
        rq = np.asarray(rq, dtype=float)
        return np.interp(rq, r, y, left=float(y[0]), right=float(y[-1]))

    return f


def _ml_gradient_map_max_envelope(
    *,
    model_obj: Any,
    X_train_ref: pd.DataFrame,
    distance_col_candidates: List[str],
    metro_stations: pd.DataFrame,
    districts_geojson_path: str,
    max_distance_m: float = 15878.0,
    grid_points: int = 200,
    r_step_m: float = 25.0,
    cmap: str = "jet",
    title: str = "Mapa gradientowa",
):
    """
    Mapa "interferencji fal" 
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import geopandas as gpd
    from shapely.geometry import Point

    if not {"lon", "lat"}.issubset(metro_stations.columns):
        raise KeyError("metro_stations musi mieć kolumny: lon, lat")

    # znalezienie właściwej kolumny odległości 
    resolved, _, _ = _resolve_columns(X_train_ref, distance_col_candidates)
    if not resolved:
        raise StepExecutionError(
            "Gradient map: nie znalazłem kolumny odległości. "
            f"Próbowałem: {distance_col_candidates}. "
            f"Dostępne (przykład): {list(X_train_ref.columns)[:50]}"
        )
    dist_col = resolved[0]

    # zbudowanie f(r) 
    r_grid = np.arange(0.0, float(max_distance_m) + float(r_step_m), float(r_step_m))
    r_grid, y_grid = _predict_curve_from_model(
        model_obj=model_obj,
        X_ref=X_train_ref,
        distance_col=dist_col,
        r_grid_m=r_grid,
    )
    f = _interp_f(r_grid, y_grid)

    # wczytanie granic Warszawy
    warsaw_districts = gpd.read_file(districts_geojson_path).to_crs(epsg=4326)

    # haversine 
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371000.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        aa = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(aa))
        return R * c

    
    lon_min, lat_min, lon_max, lat_max = warsaw_districts.total_bounds
    n = int(grid_points)
    lon_grid = np.linspace(lon_min, lon_max, n)
    lat_grid = np.linspace(lat_min, lat_max, n)
    X, Y = np.meshgrid(lon_grid, lat_grid)

    stations_lon = np.asarray(metro_stations["lon"], dtype=float)
    stations_lat = np.asarray(metro_stations["lat"], dtype=float)

    
    Z = np.full((n, n), -np.inf, dtype=float)

    for slat, slon in zip(stations_lat, stations_lon):
        r = haversine_distance(Y, X, slat, slon)  
        r = r.astype(float)
        r[r > float(max_distance_m)] = np.nan
        z_i = f(r)
        Z = np.fmax(Z, z_i)

    Z[~np.isfinite(Z)] = np.nan

    # maska Warszawy
    union_warsaw = warsaw_districts.unary_union
    try:
        from shapely import vectorized as shp_vect
        inside = shp_vect.contains(union_warsaw, X, Y)
    except Exception:
        inside = np.vectorize(lambda x, y: union_warsaw.contains(Point(float(x), float(y))))(X, Y)

    Z_masked = np.where(inside, Z, np.nan)

    # wykres
    fig, ax = plt.subplots(figsize=(10, 10))
    contour = ax.contourf(X, Y, Z_masked, levels=50, cmap=cmap, alpha=0.85)

    cbar = plt.colorbar(contour, ax=ax, label="Wartość oszacowania ceny (PLN)")
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    warsaw_districts.boundary.plot(ax=ax, color="black", linewidth=1)
    ax.scatter(metro_stations["lon"], metro_stations["lat"], color="cyan", edgecolor="black", s=20, zorder=5)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Długość geograficzna (°)")
    ax.set_ylabel("Szerokość geograficzna (°)")
    return fig, ax, {"distance_col_used": dist_col, "curve_points": int(len(r_grid))}



# ============================================================
# Rejestracja Pipeline'u
# ============================================================
def register_machine_learning_and_visualisations(
    pipe: Pipeline,
    *,
    bundle: Bundle,
    CFG: Mapping[str, Any],
    on_table: str = "data_filtered",
    artifacts_key: str = "ml_artifacts",
    out_dir: str | Path = "visualizations",
) -> Pipeline:
    """
    Rejestruje 2 kroki:
      1) ml_train -> trenuje modele z CFG['models'] i zapisuje artefakty do bundle[artifacts_key]
      2) ml_visualisations -> generuje wykresy i zapisuje do out_dir
    """
    out_dir = Path(out_dir)

    @pipe.register(
        name="ml_train",
        on_table=on_table,
        requires=(),
        produces=(),
        description="Trenuj modele + ewaluacja + ocena artefaktów w bundle",
        skip_if_all_produced_present=False,
    )
    def _ml_train(data: pd.DataFrame) -> pd.DataFrame:
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.base import clone

        exp = CFG.get("experiment", {}) or {}
        target_col = exp.get("target_col")
        if not target_col:
            raise StepExecutionError("ML: CFG['experiment']['target_col'] jest wymagane.")
        if target_col not in data.columns:
            raise StepExecutionError(f"ML: target_col='{target_col}' nie istnieje w df.")

        df = data.copy().dropna(subset=[target_col])

        preprocessor_base, colsets = _build_preprocessor(df, CFG=CFG)
        feature_cols = colsets["continuous"] + colsets["ordinal"] + colsets["binary"] + colsets["categorical"]

        df = _as_numeric(df, colsets["continuous"] + colsets["ordinal"] + colsets["binary"])
        X = df[feature_cols]
        y = pd.to_numeric(df[target_col], errors="coerce").to_numpy()

        keep = ~np.isnan(y)
        X = X.loc[keep]
        y = y[keep]

        test_size = float(exp.get("test_size", 0.2))
        random_state = int(exp.get("random_state", 42))
        cv_folds = int(exp.get("cv_folds", 5))
        scoring_list = list(exp.get("scoring", ["neg_mean_absolute_error"]))
        refit_metric = scoring_list[0]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        models_cfg = list(CFG.get("models", []) or [])
        if not models_cfg:
            raise StepExecutionError("ML: CFG['models'] jest puste.")

        fitted: Dict[str, Any] = {}
        results: List[Dict[str, Any]] = []
        cv_tables: Dict[str, pd.DataFrame] = {}
        test_preds: Dict[str, np.ndarray] = {}

        # sklearn models
        for m in models_cfg:
            name = str(m.get("name", "")).strip().lower()
            params = dict(m.get("params", {}) or {})

            if name == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                est = RandomForestRegressor(random_state=random_state, n_jobs=-1)
            elif name == "svr":
                from sklearn.svm import SVR
                est = SVR()
            elif name == "xgboost":
                try:
                    from xgboost import XGBRegressor
                except Exception as e:
                    raise StepExecutionError("ML: brak xgboost. Zainstaluj: pip install xgboost") from e
                est = XGBRegressor(random_state=random_state, verbosity=0, n_jobs=-1)
            elif name == "dnn":
                continue
            else:
                raise StepExecutionError(f"ML: nieznany model name='{name}'.")

            pre = clone(preprocessor_base)
            pipe_est = SkPipeline([("preprocess", pre), ("model", est)])
            grid_params = {f"model__{k}": v for k, v in params.items()}

            gs = GridSearchCV(
                estimator=pipe_est,
                param_grid=grid_params,
                scoring=scoring_list,
                refit=refit_metric,
                cv=cv_folds,
                n_jobs=-1,
                return_train_score=True,
            )
            gs.fit(X_train, y_train)

            best = gs.best_estimator_
            y_pred = np.asarray(best.predict(X_test), dtype=float)
            met = _metric_pack(y_test, y_pred)

            fitted[name] = best
            test_preds[name] = y_pred
            cv_tables[name] = pd.DataFrame(gs.cv_results_)

            results.append(
                {
                    "model": name,
                    "best_params": gs.best_params_,
                    "refit_metric": refit_metric,
                    "best_cv_score": float(gs.best_score_),
                    "test_mae": met["mae"],
                    "test_rmse": met["rmse"],
                    "test_mse": met["mse"],
                    "test_r2": met["r2"],
                }
            )

        # DNN
        dnn_cfg = next((x for x in models_cfg if str(x.get("name", "")).lower() == "dnn"), None)
        if dnn_cfg is not None:
            try:
                from tensorflow import keras
            except Exception as e:
                raise StepExecutionError("ML: brak tensorflow/keras dla DNN. Zainstaluj: pip install tensorflow") from e

            train_cfg = CFG.get("training", {}) or {}
            dnn_params = dict(dnn_cfg.get("params", {}) or {})

            pre_dnn = clone(preprocessor_base)
            Xtr = pre_dnn.fit_transform(X_train)
            Xte = pre_dnn.transform(X_test)

            layers = dnn_params.get("layers", [128, 64, 32])
            activation = (dnn_params.get("activation", ["relu"]) or ["relu"])[0]
            dropout = dnn_params.get("dropout", [0.0])

            if isinstance(layers, (int, float)):
                layers = [int(layers)]
            layers = [int(x) for x in layers]

            if isinstance(dropout, (int, float)):
                dropout = [float(dropout)] * len(layers)
            dropout = [float(x) for x in dropout]
            if len(dropout) < len(layers):
                dropout = dropout + [dropout[-1]] * (len(layers) - len(dropout))
            dropout = dropout[: len(layers)]

            model = keras.Sequential()
            model.add(keras.layers.Input(shape=(Xtr.shape[1],)))
            for u, dr in zip(layers, dropout):
                model.add(keras.layers.Dense(u, activation=activation))
                if dr and dr > 0:
                    model.add(keras.layers.Dropout(dr))
            model.add(keras.layers.Dense(1, activation="linear"))

            optimizer = (train_cfg.get("optimizer") or "adam")
            model.compile(optimizer=optimizer, loss="mse")

            callbacks = []
            es_cfg = (train_cfg.get("early_stopping") or {})
            if bool(es_cfg.get("enabled", True)):
                callbacks.append(
                    keras.callbacks.EarlyStopping(
                        patience=int(es_cfg.get("patience", 20)),
                        restore_best_weights=bool(es_cfg.get("restore_best_weights", True)),
                        monitor="val_loss",
                    )
                )

            model.fit(
                Xtr,
                y_train,
                epochs=int(train_cfg.get("epochs", 200)),
                batch_size=int(train_cfg.get("batch_size", 32)),
                validation_split=float(train_cfg.get("validation_split", 0.15)),
                verbose=0,
                callbacks=callbacks,
            )

            y_pred = model.predict(Xte, verbose=0).reshape(-1).astype(float)
            met = _metric_pack(y_test, y_pred)

            fitted["dnn"] = {"model": model, "preprocessor": pre_dnn}
            test_preds["dnn"] = y_pred

            results.append(
                {
                    "model": "dnn",
                    "best_params": dnn_params,
                    "refit_metric": refit_metric,
                    "best_cv_score": np.nan,
                    "test_mae": met["mae"],
                    "test_rmse": met["rmse"],
                    "test_mse": met["mse"],
                    "test_r2": met["r2"],
                }
            )

        # tabela wyników
        res_df = pd.DataFrame(results).sort_values(
            by=["test_rmse", "test_mae"], ascending=[True, True], na_position="last"
        )

        pred_grid_df = pd.DataFrame({"y_true": np.asarray(y_test, dtype=float)})
        for k, v in test_preds.items():
            v = np.asarray(v, dtype=float)
            if len(v) != len(y_test):
                raise StepExecutionError(
                    f"ML: y_pred długość != y_test dla modelu '{k}': {len(v)} vs {len(y_test)}"
                )
            pred_grid_df[f"y_pred_{k}"] = v

        df_metrics = res_df[["model", "test_mae", "test_rmse", "test_mse", "test_r2"]].copy()
        df_metrics = df_metrics.rename(columns={"test_mae": "mae", "test_rmse": "rmse", "test_mse": "mse", "test_r2": "r2"})
        df_metrics["model"] = df_metrics["model"].map(_human_model_name)

        feature_names_out = None
        for k, v in fitted.items():
            if isinstance(v, dict) and k == "dnn":
                feature_names_out = _safe_get_feature_names(v["preprocessor"])
                break
            try:
                feature_names_out = _safe_get_feature_names(v.named_steps["preprocess"])
                if feature_names_out is not None:
                    break
            except Exception:
                continue

        # ranking ważności 
        rename_dict = _make_default_feature_rename_dict()
        imp_rows = []
        for k, model_obj in fitted.items():
            df_imp = _permutation_importance_any_model(
                model_name=k,
                model_obj=model_obj if k != "dnn" else model_obj,
                X_test=X_test,
                y_test=y_test,
                feature_names_out=feature_names_out,
                n_repeats=5,
                random_state=random_state,
            )
            df_imp["feature"] = df_imp["feature"].astype(str).map(lambda x: _feature_rename(x, rename_dict))
            df_imp["model"] = _human_model_name(k)
            imp_rows.append(df_imp)

        df_importances = pd.concat(imp_rows, ignore_index=True) if imp_rows else pd.DataFrame(columns=["feature", "importance", "model"])

        
        def _make_fikcyjne_for_distance_effect(*, X_ref: pd.DataFrame, n_points: int = 400) -> pd.DataFrame:
            base = {}
            for c in X_ref.columns:
                if pd.api.types.is_numeric_dtype(X_ref[c]):
                    base[c] = float(pd.to_numeric(X_ref[c], errors="coerce").median())
                else:
                    base[c] = str(X_ref[c].mode(dropna=True).iloc[0]) if X_ref[c].notna().any() else ""

            m1 = _resolve_columns(X_ref, ["Odleglosc_metro_m1", "dist_metro_m1_m", "Odleglosc_metro_m1_m"])[0]
            m2 = _resolve_columns(X_ref, ["Odleglosc_metro_m2", "dist_metro_m2_m", "Odleglosc_metro_m2_m"])[0]
            if not m1 or not m2:
                return pd.DataFrame()

            m1c, m2c = m1[0], m2[0]
            m1_min, m1_max = (
                float(pd.to_numeric(X_ref[m1c], errors="coerce").quantile(0.02)),
                float(pd.to_numeric(X_ref[m1c], errors="coerce").quantile(0.98)),
            )
            m2_min, m2_max = (
                float(pd.to_numeric(X_ref[m2c], errors="coerce").quantile(0.02)),
                float(pd.to_numeric(X_ref[m2c], errors="coerce").quantile(0.98)),
            )

            grid_m1 = np.linspace(m1_min, m1_max, int(n_points))
            grid_m2 = np.linspace(m2_min, m2_max, int(n_points))

            rows_m1 = []
            for v in grid_m1:
                r = dict(base)
                r[m1c] = float(v)
                r[m2c] = float(base[m2c])
                rows_m1.append(r)
            df_m1 = pd.DataFrame(rows_m1)
            df_m1["log_dist_to_metro_m1"] = np.log1p(np.maximum(pd.to_numeric(df_m1[m1c], errors="coerce"), 0))
            df_m1["log_dist_to_metro_m2"] = np.nan  
            df_m1["_segment"] = "m1"

            rows_m2 = []
            for v in grid_m2:
                r = dict(base)
                r[m1c] = float(base[m1c])
                r[m2c] = float(v)
                rows_m2.append(r)
            df_m2 = pd.DataFrame(rows_m2)
            df_m2["log_dist_to_metro_m1"] = np.nan
            df_m2["log_dist_to_metro_m2"] = np.log1p(np.maximum(pd.to_numeric(df_m2[m2c], errors="coerce"), 0))
            df_m2["_segment"] = "m2"

            return pd.concat([df_m1, df_m2], ignore_index=True)

        fikcyjne = _make_fikcyjne_for_distance_effect(X_ref=X_train, n_points=400)

        # predykcje tylko dla RF i XGB 
        if not fikcyjne.empty and "random_forest" in fitted and "xgboost" in fitted:
            rf = fitted["random_forest"]
            xgb = fitted["xgboost"]

            fikcyjne["cena_predicted_m1_rf"] = np.where(
                fikcyjne["_segment"].eq("m1"),
                rf.predict(fikcyjne),
                np.nan,
            )
            fikcyjne["cena_predicted_m2_rf"] = np.where(
                fikcyjne["_segment"].eq("m2"),
                rf.predict(fikcyjne),
                np.nan,
            )
            fikcyjne["cena_predicted_m1_xgboost"] = np.where(
                fikcyjne["_segment"].eq("m1"),
                xgb.predict(fikcyjne),
                np.nan,
            )
            fikcyjne["cena_predicted_m2_xgboost"] = np.where(
                fikcyjne["_segment"].eq("m2"),
                xgb.predict(fikcyjne),
                np.nan,
            )
        else:
            fikcyjne = pd.DataFrame()

        def _poly_fit_mse(x: np.ndarray, y: np.ndarray, degree: int) -> float:
            ok = ~np.isnan(x) & ~np.isnan(y)
            x1 = x[ok]
            y1 = y[ok]
            if len(x1) < degree + 2:
                return float("nan")
            coefs = np.polyfit(x1, y1, deg=degree)
            y_hat = np.polyval(coefs, x1)
            return float(np.mean((y1 - y_hat) ** 2))

        df_stats = pd.DataFrame()
        df_params = pd.DataFrame()

        if not fikcyjne.empty:
            x_m1 = np.expm1(pd.to_numeric(fikcyjne["log_dist_to_metro_m1"], errors="coerce").to_numpy())
            x_m2 = np.expm1(pd.to_numeric(fikcyjne["log_dist_to_metro_m2"], errors="coerce").to_numpy())

            y_m1_xgb = pd.to_numeric(fikcyjne.get("cena_predicted_m1_xgboost"), errors="coerce").to_numpy()
            y_m2_xgb = pd.to_numeric(fikcyjne.get("cena_predicted_m2_xgboost"), errors="coerce").to_numpy()
            y_m1_rf = pd.to_numeric(fikcyjne.get("cena_predicted_m1_rf"), errors="coerce").to_numpy()
            y_m2_rf = pd.to_numeric(fikcyjne.get("cena_predicted_m2_rf"), errors="coerce").to_numpy()

            degrees = list(range(1, 11))
            df_stats = pd.DataFrame({"stopień": degrees})
            df_stats["MSE_M2_XGBoost"] = [_poly_fit_mse(x_m2, y_m2_xgb, d) for d in degrees]
            df_stats["MSE_M2_Rf"] = [_poly_fit_mse(x_m2, y_m2_rf, d) for d in degrees]
            df_stats["MSE_M1_XGBoost"] = [_poly_fit_mse(x_m1, y_m1_xgb, d) for d in degrees]
            df_stats["MSE_M1_Rf"] = [_poly_fit_mse(x_m1, y_m1_rf, d) for d in degrees]

            def _poly10_params(x: np.ndarray, y: np.ndarray) -> np.ndarray:
                ok = ~np.isnan(x) & ~np.isnan(y)
                x1 = x[ok]
                y1 = y[ok]
                if len(x1) < 20:
                    return np.full(11, np.nan, dtype=float)
                coefs = np.polyfit(x1, y1, deg=10)  # a0..a10
                return np.asarray(coefs, dtype=float)

            rows = {
                "M1 XGBoost": _poly10_params(x_m1, y_m1_xgb),
                "M2 RF": _poly10_params(x_m2, y_m2_rf),
            }
            df_params = pd.DataFrame.from_dict(rows, orient="index")
            df_params.columns = [f"a{i}" for i in range(11)]

        artifacts = {
            "results_table": res_df,
            "models": fitted,
            "cv_results": cv_tables,
            "pred_grid_df": pred_grid_df,
            "df_metrics": df_metrics,
            "df_importances": df_importances,
            "fikcyjne": fikcyjne,
            "df_stats": df_stats,
            "df_params": df_params,
            "X_train": X_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names_out": feature_names_out,
            "colsets": colsets,
            "experiment": dict(exp),
        }
        bundle.set(artifacts_key, artifacts)
        return data

    @pipe.register(
        name="ml_visualisations",
        on_table=on_table,
        requires=(artifacts_key,),
        produces=(),
        description="Create ML plots and save to out_dir",
        skip_if_all_produced_present=False,
    )
    def _ml_visualisations(data: pd.DataFrame) -> pd.DataFrame:
        import matplotlib.pyplot as plt

        art = bundle.get(artifacts_key)
        if not isinstance(art, dict):
            raise StepExecutionError("ML viz: bundle[artifacts_key] nie jest dict (brak artifacts?).")

        out_dir.mkdir(parents=True, exist_ok=True)
        figs: Dict[str, str] = {}


        viz_cfg = CFG.get("visualisations", {}) or {}
        show_plots = bool(viz_cfg.get("show_plots")) if ("show_plots" in viz_cfg) else _in_notebook()

        # (1) Słupki z metrykami
        df_metrics: pd.DataFrame = art.get("df_metrics")
        if df_metrics is None or df_metrics.empty:
            raise StepExecutionError("ML viz: brak df_metrics w artifacts.")
        fig, _ = ml_metrics_bar_models(df_metrics)
        figs["01_metrics_bar_models"] = _save_fig(fig, out_dir / "01_metrics_bar_models.png")
        _maybe_show_matplotlib(fig, enabled=show_plots)
        plt.close(fig)

        # (2) KDE grid
        pred_grid_df: pd.DataFrame = art.get("pred_grid_df")
        if pred_grid_df is None or pred_grid_df.empty:
            raise StepExecutionError("ML viz: brak pred_grid_df w artifacts.")
        fig, _ = ml_true_vs_pred_kde_grid(pred_grid_df, ncols=2)
        figs["02_true_vs_pred_kde_grid"] = _save_fig(fig, out_dir / "02_true_vs_pred_kde_grid.png")
        _maybe_show_matplotlib(fig, enabled=show_plots)
        plt.close(fig)

        # (3) Ranking Ważności (RF vs XGB)
        df_importances: pd.DataFrame = art.get("df_importances")
        if df_importances is None or df_importances.empty:
            raise StepExecutionError("ML viz: brak df_importances w artifacts.")
        fig, _ = ml_feature_importance_rf_vs_xgboost(df_importances, top_n=20)
        figs["03_feature_importance_rf_vs_xgboost"] = _save_fig(fig, out_dir / "03_feature_importance_rf_vs_xgboost.png")
        _maybe_show_matplotlib(fig, enabled=show_plots)
        plt.close(fig)

        # (4) Błąd średniokwadratowy w zależności od stopnia wielomianu
        df_stats: pd.DataFrame = art.get("df_stats")
        if df_stats is None or df_stats.empty:
            raise StepExecutionError("ML viz: brak df_stats (do poly-degree MSE).")
        fig, _ = ml_poly_degree_mse_panel(df_stats)
        figs["04_poly_degree_mse_panel"] = _save_fig(fig, out_dir / "04_poly_degree_mse_panel.png")
        _maybe_show_matplotlib(fig, enabled=show_plots)
        plt.close(fig)

        # (5) Interpolacja dystansu na danych fikcyjnych
        fikcyjne: pd.DataFrame = art.get("fikcyjne")
        if fikcyjne is None or fikcyjne.empty:
            raise StepExecutionError("ML viz: brak fikcyjne (do distance effect).")
        fig, _ = ml_distance_effect_poly10_panel(fikcyjne)
        figs["05_distance_effect_poly10_panel"] = _save_fig(fig, out_dir / "05_distance_effect_poly10_panel.png")
        _maybe_show_matplotlib(fig, enabled=show_plots)
        plt.close(fig)

        
        # (6-9) Mapy gradientowe
        districts_geojson_path = viz_cfg.get("districts_geojson_path")
        if not districts_geojson_path:
            for cand in [
                "geojson/warsaw_districts.geojson",
                "warsaw_districts.geojson",
                "geojson/Administracyjne_Warszawa_Dzielnice.geojson",
                "Administracyjne_Warszawa_Dzielnice.geojson",
            ]:
                if Path(cand).exists():
                    districts_geojson_path = cand
                    break

        # stacje metra 
        metro_stations: Optional[pd.DataFrame] = None

        for key in ("metro4", "metro", "metro_stations"):
            try:
                tmp = bundle.get(key)
            except Exception:
                tmp = None
            if isinstance(tmp, pd.DataFrame) and {"lon", "lat"}.issubset(tmp.columns) and len(tmp):
                metro_stations = tmp[["lon", "lat"]].copy()
                break

        def _load_points_from_geojson(path: str | Path) -> pd.DataFrame:
            import geopandas as gpd

            gdf = gpd.read_file(str(path))
            try:
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=4326, allow_override=True)
                gdf = gdf.to_crs(epsg=4326)
            except Exception:
                pass

            if "lon" in gdf.columns and "lat" in gdf.columns:
                dfp = gdf[["lon", "lat"]].copy()
            else:
                if gdf.geometry is None:
                    raise StepExecutionError(f"GeoJSON '{path}' nie ma geometrii Point ani kolumn lon/lat.")
                gg = gdf[gdf.geometry.notna()].copy()
                dfp = pd.DataFrame({"lon": gg.geometry.x.astype(float), "lat": gg.geometry.y.astype(float)})

            return dfp.dropna(subset=["lon", "lat"])

        if metro_stations is None:
            metro_stations_path = viz_cfg.get("metro_stations_path")
            if metro_stations_path:
                pth = Path(metro_stations_path)
                if pth.suffix.lower() in (".geojson", ".json"):
                    metro_stations = _load_points_from_geojson(pth)
                else:
                    metro_stations = pd.read_csv(pth)
                    if not {"lon", "lat"}.issubset(metro_stations.columns):
                        raise StepExecutionError("metro_stations_path musi mieć kolumny lon, lat.")

        if metro_stations is None:
            for cand in ["geojson/metro.geojson", "metro.geojson"]:
                if Path(cand).exists():
                    metro_stations = _load_points_from_geojson(cand)
                    break
        
        m4_geo = Path("geojson/metro_m4_stations.geojson")
        if m4_geo.exists():
            metro_stations = _load_points_from_geojson(m4_geo)


        if districts_geojson_path and isinstance(metro_stations, pd.DataFrame) and not metro_stations.empty:
            try:
                models: Dict[str, Any] = art.get("models") or {}
                X_train_ref: pd.DataFrame = art.get("X_train")
                if X_train_ref is None or not isinstance(X_train_ref, pd.DataFrame) or X_train_ref.empty:
                    raise StepExecutionError("ML viz: brak X_train w artifacts (potrzebne do 'typowego mieszkania').")

    
                xgb = models.get("xgboost")
                rf = models.get("random_forest")
                if xgb is None or rf is None:
                    raise StepExecutionError("ML viz: brak modelu 'xgboost' i/lub 'random_forest' w artifacts.")

                max_distance_m = float(viz_cfg.get("max_distance_m", 15878.0))
                grid_points = int(viz_cfg.get("grid_points", 200))
                r_step_m = float(viz_cfg.get("r_step_m", 25.0))

                m1_candidates = ["Odleglosc_metro_m1", "dist_metro_m1_m", "Odleglosc_metro_m1_m"]
                m2_candidates = ["Odleglosc_metro_m2", "dist_metro_m2_m", "Odleglosc_metro_m2_m"]

                # M1 XGBoost
                fig, _, info = _ml_gradient_map_max_envelope(
                    model_obj=xgb,
                    X_train_ref=X_train_ref,
                    distance_col_candidates=m1_candidates,
                    metro_stations=metro_stations,
                    districts_geojson_path=str(districts_geojson_path),
                    max_distance_m=max_distance_m,
                    grid_points=grid_points,
                    r_step_m=r_step_m,
                    cmap="jet",
                    title="Mapa gradientowa potencjalnych zmian cen \n(M1 – XGBoost)",
                )
                figs["06_gradient_map_max_m1_xgboost"] = _save_fig(fig, out_dir / "06_gradient_map_max_m1_xgboost.png")
                _maybe_show_matplotlib(fig, enabled=show_plots)
                plt.close(fig)

                # M1 RF
                fig, _, info = _ml_gradient_map_max_envelope(
                    model_obj=rf,
                    X_train_ref=X_train_ref,
                    distance_col_candidates=m1_candidates,
                    metro_stations=metro_stations,
                    districts_geojson_path=str(districts_geojson_path),
                    max_distance_m=max_distance_m,
                    grid_points=grid_points,
                    r_step_m=r_step_m,
                    cmap="viridis",
                    title="Mapa gradientowa potencjalnych zmian cen \n(M1 – Random Forest)",
                )
                figs["07_gradient_map_max_m1_rf"] = _save_fig(fig, out_dir / "07_gradient_map_max_m1_rf.png")
                _maybe_show_matplotlib(fig, enabled=show_plots)
                plt.close(fig)

                # M2 XGBoost
                fig, _, info = _ml_gradient_map_max_envelope(
                    model_obj=xgb,
                    X_train_ref=X_train_ref,
                    distance_col_candidates=m2_candidates,
                    metro_stations=metro_stations,
                    districts_geojson_path=str(districts_geojson_path),
                    max_distance_m=max_distance_m,
                    grid_points=grid_points,
                    r_step_m=r_step_m,
                    cmap="jet",
                    title="Mapa gradientowa potencjalnych zmian cen \n(M2 – XGBoost)",
                )
                figs["08_gradient_map_max_m2_xgboost"] = _save_fig(fig, out_dir / "08_gradient_map_max_m2_xgboost.png")
                _maybe_show_matplotlib(fig, enabled=show_plots)
                plt.close(fig)

                # M2 RF
                fig, _, info = _ml_gradient_map_max_envelope(
                    model_obj=rf,
                    X_train_ref=X_train_ref,
                    distance_col_candidates=m2_candidates,
                    metro_stations=metro_stations,
                    districts_geojson_path=str(districts_geojson_path),
                    max_distance_m=max_distance_m,
                    grid_points=grid_points,
                    r_step_m=r_step_m,
                    cmap="viridis",
                    title="Mapa gradientowa potencjalnych zmian cen \n(M2 – Random Forest)",
                )
                figs["09_gradient_map_max_m2_rf"] = _save_fig(fig, out_dir / "09_gradient_map_max_m2_rf.png")
                _maybe_show_matplotlib(fig, enabled=show_plots)
                plt.close(fig)

            except Exception as e:
                figs["06_09_gradient_maps_max_skipped"] = f"SKIPPED: {type(e).__name__}: {e}"
        else:
            figs["06_09_gradient_maps_max_skipped"] = (
                "SKIPPED: brakuje danych do map. "
                "Ustaw CFG['visualisations']['districts_geojson_path'] albo trzymaj geojson/warsaw_districts.geojson, "
                "oraz dostarcz metro (bundle['metro4'] z lon/lat lub metro.geojson)."
            )

        art["figures"] = figs
        bundle.set(artifacts_key, art)
        return data

    return pipe