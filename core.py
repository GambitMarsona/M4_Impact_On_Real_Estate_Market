# core.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd


class StepExecutionError(RuntimeError):
    """Błąd wykonania kroku w pipelinie (tak jak w Twoim DX)."""
    pass


class Bundle:
    """
    Prosty store na DataFrame'y i inne obiekty.

    Użycie:
        bundle = Bundle()
        bundle.set("data", df)
        df2 = bundle.get("data")
    """
    def __init__(self, **initial: Any) -> None:
        self._store: Dict[str, Any] = dict(initial)

    def has(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def keys(self) -> List[str]:
        return list(self._store.keys())

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._store)


@dataclass
class PipelineStep:
    name: str
    on_table: str
    requires: Tuple[str, ...]
    produces: Tuple[str, ...]
    description: str
    skip_if_all_produced_present: bool
    func: Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class PipelineLogEntry:
    name: str
    started_at: datetime
    finished_at: datetime
    on_table: str
    n_rows_before: int
    n_cols_before: int
    n_rows_after: int
    n_cols_after: int
    description: str


class Pipeline:
    """
    Minimalny pipeline w stylu tego z DX:
    - pipe.register(...) jako dekorator
    - pipe.run() wykonuje kroki po kolei na Bundle
    """
    def __init__(self, bundle: Bundle):
        self.bundle = bundle
        self._steps: List[PipelineStep] = []
        self._log: List[PipelineLogEntry] = []

    # -------- rejestracja kroków --------
    def register(
        self,
        *,
        name: str,
        on_table: str,
        requires: Sequence[str] | None = None,
        produces: Sequence[str] | None = None,
        description: str = "",
        skip_if_all_produced_present: bool = True,
    ):
        """
        Użycie:
            @pipe.register(
                name="features_dx_compute",
                on_table="data",
                requires=("page_location_series",),
                produces=DX_COLS,
                ...
            )
            def _dx(data: pd.DataFrame) -> pd.DataFrame:
                ...
        """

        requires = tuple(requires or ())
        produces = tuple(produces or ())

        def decorator(func: Callable[[pd.DataFrame], pd.DataFrame]):
            step = PipelineStep(
                name=name,
                on_table=on_table,
                requires=requires,
                produces=produces,
                description=description,
                skip_if_all_produced_present=skip_if_all_produced_present,
                func=func,
            )
            self._steps.append(step)
            return func

        return decorator

    # -------- wykonanie pipeline'u --------
    def run(self, *, until: Optional[str] = None) -> Bundle:
        """
        Uruchamia wszystkie kroki po kolei.
        Jeśli podasz until="nazwa_kroku", zatrzyma się po tym kroku.
        """
        for step in self._steps:
            # jeśli jest until i jesteśmy "po", wychodzimy
            # (ale krok z nazwą == until jeszcze się wykona)
            df = self.bundle.get(step.on_table)
            if df is None:
                raise StepExecutionError(
                    f"Step '{step.name}' oczekuje tabeli '{step.on_table}' w bundle, ale jej nie znaleziono."
                )
            if not isinstance(df, pd.DataFrame):
                raise StepExecutionError(
                    f"Step '{step.name}' oczekuje DataFrame w bundle['{step.on_table}'], dostał: {type(df)}"
                )

            # sprawdź wymagane tabele w bundle
            missing_requires = [r for r in step.requires if not self.bundle.has(r)]
            if missing_requires:
                raise StepExecutionError(
                    f"Step '{step.name}' wymaga w bundle: {missing_requires}, ale ich nie ma."
                )

            # opcjonalne skipowanie, jeśli wszystkie kolumny już są
            if step.skip_if_all_produced_present and step.produces:
                if all(col in df.columns for col in step.produces):
                    continue

            n_rows_before, n_cols_before = df.shape
            started = datetime.now()
            out = step.func(df)
            finished = datetime.now()

            if not isinstance(out, pd.DataFrame):
                raise StepExecutionError(
                    f"Step '{step.name}' powinien zwrócić DataFrame, dostał: {type(out)}"
                )

            n_rows_after, n_cols_after = out.shape
            self.bundle.set(step.on_table, out)

            self._log.append(
                PipelineLogEntry(
                    name=step.name,
                    started_at=started,
                    finished_at=finished,
                    on_table=step.on_table,
                    n_rows_before=n_rows_before,
                    n_cols_before=n_cols_before,
                    n_rows_after=n_rows_after,
                    n_cols_after=n_cols_after,
                    description=step.description,
                )
            )

            if until is not None and step.name == until:
                break

        return self.bundle


    def log_as_dataframe(self) -> pd.DataFrame:
        if not self._log:
            return pd.DataFrame(
                columns=[
                    "name",
                    "started_at",
                    "finished_at",
                    "on_table",
                    "n_rows_before",
                    "n_cols_before",
                    "n_rows_after",
                    "n_cols_after",
                    "description",
                ]
            )
        return pd.DataFrame([vars(e) for e in self._log])
