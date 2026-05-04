"""Rolling early-warning style statistics: mean, variance, skew over a window on time series (e.g. I)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

__all__ = [
    "rolling_ews",
    "ews_for_i_column",
    "filter_runs",
    "pointwise_mean_std",
    "iter_simulation_ews",
    "stack_ews_for_runs",
    "mean_argmax_time_index",
    "plot_ews_grid_by_lam",
    "plot_ews_mean_band",
]


def rolling_ews(series, window: int = 10) -> pd.DataFrame:
    """
    Rolling mean, variance (ddof=0), and sample skewness (bias=False) with full windows only.
    """
    w = max(1, int(window))
    s = series if isinstance(series, pd.Series) else pd.Series(series, copy=False)
    roll = s.rolling(w, min_periods=w)
    return pd.DataFrame(
        {
            "mean": roll.mean(),
            "var": roll.var(ddof=0),
            "skew": roll.apply(
                lambda x: stats.skew(np.asarray(x, dtype=float), bias=False), raw=True
            ),
        }
    )


def ews_for_i_column(
    path: Path, window: int = 10, column: str = "I"
) -> pd.DataFrame:
    df = pd.read_csv(path)
    return rolling_ews(df[column], window=window)


def filter_runs(
    runs: pd.DataFrame,
    graph_kind: str,
    lam_values: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    m = runs["graph_kind"] == graph_kind
    out = runs.loc[m].copy()
    if lam_values is not None:
        lam_set = {float(x) for x in lam_values}
        out = out.loc[out["lam"].map(lambda x: float(x) in lam_set)]
    return out.reset_index(drop=True)


def _resolve_path(data_root: Path, relative_path: str) -> Path:
    p = (data_root / relative_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def mean_argmax_time_index(
    runs: pd.DataFrame,
    data_root: Path,
    column: str = "I",
) -> float:
    """Mean over runs of the time index (day) where `column` attains its maximum (e.g. peak I)."""
    if len(runs) == 0:
        return float("nan")
    peaks: List[int] = []
    for _, row in runs.iterrows():
        path = _resolve_path(data_root, str(row["relative_path"]))
        df = pd.read_csv(path)
        arr = np.asarray(df[column].to_numpy(), dtype=float)
        peaks.append(int(np.argmax(arr)))
    return float(np.mean(peaks))


def iter_simulation_ews(
    runs: pd.DataFrame,
    data_root: Path,
    window: int = 10,
    column: str = "I",
) -> Iterator[Tuple[pd.Series, pd.DataFrame]]:
    for _, row in runs.iterrows():
        path = _resolve_path(data_root, str(row["relative_path"]))
        ews = ews_for_i_column(path, window=window, column=column)
        yield row, ews


def pointwise_mean_std(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """mat shape (n_sim, T) — mean and std over simulations at each time index."""
    m = np.nanmean(mat, axis=0)
    s = np.nanstd(mat, axis=0, ddof=0)
    return m, s


def _stack_ews_optimized(
    runs: pd.DataFrame,
    data_root: Path,
    window: int,
) -> np.ndarray:
    n_runs = len(runs)
    if n_runs == 0:
        return np.empty((0, 0, 3))
    t_len = 0
    arrs: List[np.ndarray] = []
    for _, row in runs.iterrows():
        path = _resolve_path(data_root, str(row["relative_path"]))
        ews = ews_for_i_column(path, window=window, column="I")
        a = np.column_stack(
            [ews["mean"].to_numpy(), ews["var"].to_numpy(), ews["skew"].to_numpy()]
        )
        arrs.append(a)
        t_len = a.shape[0]
    return np.stack(arrs, axis=0)


def stack_ews_for_runs(
    runs: pd.DataFrame,
    data_root: Path,
    window: int = 10,
) -> np.ndarray:
    """
    Shape (n_sim, T, 3) with channels mean, var, skew of rolling EWS on I.
    """
    return _stack_ews_optimized(runs, data_root, window)


def plot_ews_grid_by_lam(
    runs: pd.DataFrame,
    data_root: Path,
    window: int = 10,
    graph_kind: str = "barabasi_albert_graph",
    lam_values: Optional[Sequence[float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show_mean_peak: bool = True,
) -> plt.Figure:
    """
    Three rows: rolling mean, var, skew. One column per distinct lam.
    Each curve is the pointwise mean of EWS over all tau/seed for that lam.
    """
    sub = filter_runs(runs, graph_kind, lam_values=lam_values)
    lams = sorted(sub["lam"].unique().tolist(), key=lambda x: float(x))
    n_lam = len(lams)
    if n_lam == 0:
        fig, _ = plt.subplots()
        return fig
    w_fig = max(3.5 * n_lam, 8)
    h_fig = 9.0
    if figsize is None:
        figsize = (w_fig, h_fig)
    fig, axes = plt.subplots(3, n_lam, figsize=figsize, sharex="col", sharey="row",
                             squeeze=False)
    metric_names = ("mean(I)", "var(I)", "skew(I)")
    t_axis = None
    for j, lam in enumerate(lams):
        part = sub[sub["lam"] == lam]
        st = stack_ews_for_runs(part, data_root, window=window)
        if st.size == 0:
            continue
        t_len = st.shape[1]
        t_axis = np.arange(t_len)
        for r in range(3):
            col = st[:, :, r]
            m, _ = pointwise_mean_std(col)
            axes[r, j].plot(t_axis, m, color="C0", lw=1.2)
        t_peak_mean = mean_argmax_time_index(part, data_root)
        if show_mean_peak and np.isfinite(t_peak_mean):
            for r in range(3):
                axes[r, j].axvline(
                    t_peak_mean,
                    color="0.3",
                    ls="--",
                    lw=1.0,
                    alpha=0.85,
                    zorder=5,
                )
        axes[0, j].set_title(f"lam={gfmt(lam)}")
    for r in range(3):
        axes[r, 0].set_ylabel(metric_names[r])
    for j in range(n_lam):
        axes[2, j].set_xlabel("t (day index)")
    fig.suptitle(
        f"EWS (window={window}) - {graph_kind}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def gfmt(x: float) -> str:
    y = float(x)
    if abs(y - int(y)) < 1e-9:
        return str(int(y))
    return f"{y:g}"


def plot_ews_mean_band(
    runs: pd.DataFrame,
    data_root: Path,
    window: int = 10,
    graph_kind: str = "barabasi_albert_graph",
    lam_values: Optional[Sequence[float]] = None,
    show_mean_peak: bool = True,
) -> plt.Figure:
    """
    One figure, three rows: pointwise mean ± std of rolling EWS across all simulations
    in the filtered runs table.
    """
    sub = filter_runs(runs, graph_kind, lam_values=lam_values)
    st = stack_ews_for_runs(sub, data_root, window=window)
    n_runs, t_len, _ = st.shape
    if n_runs == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No runs match filter", ha="center", va="center")
        ax.axis("off")
        return fig
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    names = ("mean EWS of I (rolling mean)", "var EWS of I (rolling var)", "skew EWS of I (rolling skew)")
    t_axis = np.arange(t_len) if t_len else np.array([])
    t_peak_mean = mean_argmax_time_index(sub, data_root)
    for r in range(3):
        ax = axes[r]
        col = st[:, :, r]
        m, s = pointwise_mean_std(col)
        ax.fill_between(
            t_axis, m - s, m + s, color="C0", alpha=0.25, label="mean ± std" if r == 0 else None
        )
        ax.plot(t_axis, m, color="C0", lw=1.2, label="mean" if r == 0 else None)
        if show_mean_peak and np.isfinite(t_peak_mean):
            ax.axvline(
                t_peak_mean,
                color="0.3",
                ls="--",
                lw=1.0,
                alpha=0.85,
                zorder=5,
            )
        ax.set_ylabel(names[r], fontsize=9)
    axes[-1].set_xlabel("t (day index)")
    title_lam = "all lam" if lam_values is None else f"lam in {list(lam_values)}"
    fig.suptitle(
        f"Mean ± std of EWS over n={n_runs} sims ({title_lam}) — {graph_kind}, window={window}",
        fontsize=11,
    )
    if n_runs and axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# #region agent log
def _agent_log_ews_load() -> None:
    import json
    import time
    import inspect

    _log = Path(__file__).resolve().parent / "debug-e3ee3b.log"
    try:
        _params = list(inspect.signature(plot_ews_grid_by_lam).parameters.keys())
        with open(_log, "a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "e3ee3b",
                        "hypothesisId": "H1_stale_or_shadow",
                        "location": "ews.py:_agent_log_ews_load",
                        "message": "module import snapshot",
                        "data": {
                            "ews___file__": str(Path(__file__).resolve()),
                            "plot_ews_grid_by_lam_param_names": _params,
                            "has_show_mean_peak": "show_mean_peak" in _params,
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except OSError:
        pass


_agent_log_ews_load()
# #endregion
