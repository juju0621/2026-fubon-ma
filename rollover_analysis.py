import json
import logging
import warnings
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no GUI pop-ups
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Dual-handler logger: file=DEBUG, console=INFO."""
    Path(log_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"fitx_combined_{ts}.log"

    logger = logging.getLogger("FITXCombined")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    ))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log → {log_file}")
    return logger


def configure_plot_style():
    plt.rcParams.update({
        "figure.dpi": 130, "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA", "axes.grid": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.25, "grid.linestyle": "--",
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "legend.fontsize": 9, "lines.linewidth": 1.4,
    })


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContractInfo:
    """Parsed contract metadata from Seccode (e.g. FITX_202406 or FITX_202406/202407)."""
    seccode: str
    is_roll_spread: bool = False
    front_month: Optional[str] = None   # YYYYMM
    back_month: Optional[str] = None
    expiry_date: Optional[pd.Timestamp] = None

    @staticmethod
    def third_wednesday(year: int, month: int) -> pd.Timestamp:
        """FITX settlement = 3rd Wednesday of the contract month."""
        first = pd.Timestamp(year, month, 1)
        offset = (2 - first.weekday()) % 7      # days to first Wednesday
        return first + pd.Timedelta(days=offset) + pd.Timedelta(weeks=2)

    @classmethod
    def parse(cls, seccode: str) -> "ContractInfo":
        info = cls(seccode=seccode)
        body = seccode.replace("FITX_", "")
        if "/" in body:
            parts = body.split("/")
            info.is_roll_spread = True
            info.front_month = parts[0].strip()
            info.back_month = parts[1].strip()
        else:
            info.is_roll_spread = False
            info.front_month = body.strip()
        if info.front_month and len(info.front_month) == 6:
            try:
                info.expiry_date = cls.third_wednesday(
                    int(info.front_month[:4]), int(info.front_month[4:6])
                )
            except (ValueError, IndexError):
                pass
        return info


@dataclass
class CleaningReport:
    total_rows_raw: int = 0
    duplicates_removed: int = 0
    null_rows_removed: int = 0
    price_outliers_flagged: int = 0
    volume_outliers_flagged: int = 0
    negative_values_fixed: int = 0
    ohlc_violations_fixed: int = 0
    total_rows_clean: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "DATA CLEANING REPORT",
            "=" * 60,
            f"  Raw rows:                {self.total_rows_raw:>10,}",
            f"  Duplicates removed:      {self.duplicates_removed:>10,}",
            f"  Null rows removed:       {self.null_rows_removed:>10,}",
            f"  Price outliers flagged:  {self.price_outliers_flagged:>10,}",
            f"  Volume outliers flagged: {self.volume_outliers_flagged:>10,}",
            f"  Negative values fixed:   {self.negative_values_fixed:>10,}",
            f"  OHLC violations fixed:   {self.ohlc_violations_fixed:>10,}",
            f"  Clean rows:              {self.total_rows_clean:>10,}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

class FITXDataLoader:
    """Load, validate, and clean FITX minute-bar CSV data."""

    PRICE_COLS  = ["Px_O", "Px_H", "Px_L", "Px_C"]
    VOLUME_COLS = ["Trade_Vol", "Trade_Cnt"]

    def __init__(self, filepath: str, logger: logging.Logger):
        self.filepath = filepath
        self.log = logger
        self.report = CleaningReport()

    def load(self) -> pd.DataFrame:
        self.log.info(f"Loading data from {self.filepath}")
        df = pd.read_csv(
            self.filepath,
            parse_dates=["Txdate"],
            dtype={"Seccode": str, "BTime": str, "ETime": str},
        )
        self.report.total_rows_raw = len(df)
        self.log.info(f"Loaded {len(df):,} rows | cols: {list(df.columns)}")
        self.log.debug(
            f"Date range: {df['Txdate'].min().date()} → {df['Txdate'].max().date()}"
        )
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._remove_duplicates(df)
        df = self._handle_nulls(df)
        df = self._fix_negative_values(df)
        df = self._fix_ohlc_consistency(df)
        df = self._flag_price_outliers(df)
        df = self._flag_volume_outliers(df)
        df = self._enrich_contract_metadata(df)
        self.report.total_rows_clean = len(df)
        self.log.info(self.report.summary())
        return df

    # ── private cleaning steps ────────────────────────────────────────────────

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        # BTime/ETime may be missing for some rows; guard with subset that exists
        key_cols = [c for c in ["Txdate", "Seccode", "BTime", "ETime"] if c in df.columns]
        df = df.drop_duplicates(subset=key_cols)
        self.report.duplicates_removed = before - len(df)
        self.log.debug(f"Duplicates removed: {self.report.duplicates_removed}")
        return df.reset_index(drop=True)

    def _handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df[self.PRICE_COLS].isnull().any(axis=1)
        self.report.null_rows_removed = int(mask.sum())
        df = df[~mask].copy()
        for col in self.VOLUME_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df.reset_index(drop=True)

    def _fix_negative_values(self, df: pd.DataFrame) -> pd.DataFrame:
        count = 0
        for col in self.PRICE_COLS + self.VOLUME_COLS:
            if col not in df.columns:
                continue
            neg = df[col] < 0
            n = int(neg.sum())
            if n:
                self.log.warning(f"'{col}': {n} negative values corrected")
                df.loc[neg, col] = np.nan if col in self.PRICE_COLS else 0
                count += n
        self.report.negative_values_fixed = count
        df = df[df[self.PRICE_COLS].notnull().all(axis=1)].reset_index(drop=True)
        return df

    def _fix_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        oc_max = df[["Px_O", "Px_C"]].max(axis=1)
        oc_min = df[["Px_O", "Px_C"]].min(axis=1)
        h_viol = df["Px_H"] < oc_max
        l_viol = df["Px_L"] > oc_min
        count = int((h_viol | l_viol).sum())
        if count:
            self.log.warning(f"OHLC violations fixed: {count}")
            df.loc[h_viol, "Px_H"] = oc_max[h_viol]
            df.loc[l_viol, "Px_L"] = oc_min[l_viol]
        self.report.ohlc_violations_fixed = count
        return df

    def _flag_price_outliers(self, df: pd.DataFrame, z: float = 5.0) -> pd.DataFrame:
        df["is_price_outlier"] = False
        for sc, grp in df.groupby("Seccode"):
            if len(grp) < 30:
                continue
            ret = grp["Px_C"].pct_change().dropna()
            if ret.std() == 0:
                continue
            idx = ret.index[np.abs(stats.zscore(ret)) > z]
            df.loc[idx, "is_price_outlier"] = True
        self.report.price_outliers_flagged = int(df["is_price_outlier"].sum())
        return df

    def _flag_volume_outliers(self, df: pd.DataFrame, z: float = 5.0) -> pd.DataFrame:
        df["is_vol_outlier"] = False
        for sc, grp in df.groupby("Seccode"):
            if len(grp) < 30 or grp["Trade_Vol"].std() == 0:
                continue
            idx = grp["Trade_Vol"].index[np.abs(stats.zscore(grp["Trade_Vol"])) > z]
            df.loc[idx, "is_vol_outlier"] = True
        self.report.volume_outliers_flagged = int(df["is_vol_outlier"].sum())
        return df

    def _enrich_contract_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        cache = {sc: ContractInfo.parse(sc) for sc in df["Seccode"].unique()}
        df["is_roll_spread"] = df["Seccode"].map(lambda s: cache[s].is_roll_spread)
        df["front_month"]    = df["Seccode"].map(lambda s: cache[s].front_month)
        df["back_month"]     = df["Seccode"].map(lambda s: cache[s].back_month)
        df["expiry_date"]    = df["Seccode"].map(lambda s: cache[s].expiry_date)
        df["days_to_expiry"] = (df["expiry_date"] - df["Txdate"]).dt.days
        return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — ROLL PANEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class RollPanelBuilder:
    """
    Build a daily near-month vs next-month panel:
      - spread (front − back close)
      - volume ratio (back / front)
      - per-cycle metadata for all ~28 monthly cycles in the dataset
    Also retains raw spread contract data (with BTime) for intraday use.
    """

    DTE_BINS   = [0, 3, 6, 9, 13, 18, 26, 40]
    DTE_LABELS = ["T-0~2","T-3~5","T-6~8","T-9~12","T-13~17","T-18~25","T-26~40"]

    def __init__(self, df: pd.DataFrame, logger: logging.Logger):
        self.log = logger
        self.outr = df[~df["is_roll_spread"] & df[["Px_O","Px_H","Px_L","Px_C"]].gt(0).all(axis=1)].copy()
        # Keep raw spread bars (includes BTime) for intraday analysis
        self.sprd_raw = df[df["is_roll_spread"]].copy()

    @staticmethod
    def _next_yyyymm(s: str) -> str:
        y, m = int(s[:4]), int(s[4:6])
        m += 1
        if m > 12:
            m, y = 1, y + 1
        return f"{y}{m:02d}"

    def build(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
          panel      — daily near/next panel with spread & vol_ratio
          sprd_1m    — raw minute-bar 1-month spread contracts (BTime intact)
        """
        self.log.info("Building daily roll panel...")

        # ── daily outright aggregates ─────────────────────────────────────────
        daily = (
            self.outr.groupby(["Txdate", "front_month", "expiry_date", "days_to_expiry"])
            .agg(close=("Px_C","last"), volume=("Trade_Vol","sum"),
                 cnt=("Trade_Cnt","sum"))
            .reset_index()
            .rename(columns={"expiry_date":"expiry","days_to_expiry":"dte"})
        )
        daily = daily[daily["dte"] >= 0]

        # ── pair near / next per trading date ─────────────────────────────────
        rows = []
        for date, grp in daily.groupby("Txdate"):
            months = grp.sort_values("dte")
            if len(months) < 2:
                continue
            near, nxt = months.iloc[0], months.iloc[1]
            vol_ratio = nxt["volume"] / near["volume"] if near["volume"] > 0 else np.nan
            rows.append({
                "date":       date,
                "near_month": near["front_month"],
                "next_month": nxt["front_month"],
                "expiry":     near["expiry"],
                "dte":        near["dte"],
                "near_close": near["close"],
                "next_close": nxt["close"],
                "spread":     near["close"] - nxt["close"],
                "near_vol":   near["volume"],
                "next_vol":   nxt["volume"],
                "vol_ratio":  vol_ratio,
                "near_cnt":   near["cnt"],
                "next_cnt":   nxt["cnt"],
            })

        panel = pd.DataFrame(rows)
        panel["date"] = pd.to_datetime(panel["date"])
        panel["dte_bin"] = pd.cut(
            panel["dte"], bins=self.DTE_BINS, labels=self.DTE_LABELS, right=True
        )

        # ── 1-month raw spread contracts (keep BTime) ─────────────────────────
        self.sprd_raw["expected_back"] = self.sprd_raw["front_month"].apply(
            self._next_yyyymm
        )
        sprd_1m = self.sprd_raw[
            self.sprd_raw["back_month"] == self.sprd_raw["expected_back"]
        ].copy()

        n_cycles = panel["near_month"].nunique()
        self.log.info(f"Roll cycles: {n_cycles} | Panel rows: {len(panel):,}")
        self.log.info(f"1-month raw spread bars: {len(sprd_1m):,}")
        return panel, sprd_1m


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — ROLL YIELD ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class RollYieldAnalyzer:
    """
    Compute daily roll yield from outright price differentials.

    Roll yield = (Front − Back) / Front × 100 %
      Positive (backwardation) → long rollers collect a credit
      Negative (contango)      → short rollers collect a credit
    """

    def __init__(self, panel: pd.DataFrame, logger: logging.Logger):
        self.panel = panel.copy()
        self.log = logger

    def compute(self) -> pd.DataFrame:
        self.log.info("Computing roll yield metrics...")
        df = self.panel.copy()
        df["roll_yield_pct"] = (
            (df["near_close"] - df["next_close"]) / df["near_close"] * 100
        )
        df["roll_yield_ann"] = df["roll_yield_pct"] * 12
        sp = df["spread"]
        self.log.info(
            f"Roll yield — mean spread: {sp.mean():.1f} pts | "
            f"median: {sp.median():.1f} pts | "
            f"backwardation: {(sp>0).mean()*100:.1f}% | "
            f"contango: {(sp<0).mean()*100:.1f}%"
        )
        return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — LIQUIDITY ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class LiquidityAnalyzer:
    """
    Analyze liquidity migration (volume crossover) from near to back month.

    Key output: per-cycle DTE at which back-month volume first exceeds near-month.
    """

    def __init__(self, panel: pd.DataFrame, logger: logging.Logger):
        self.panel = panel.copy()
        self.log = logger

    def crossover_dtes(self) -> pd.DataFrame:
        """Return per-cycle DTE where back-month volume ≥ front-month volume."""
        rows = []
        for cyc, grp in self.panel.groupby("near_month"):
            sub = grp[grp["vol_ratio"] >= 1.0]
            if not sub.empty:
                rows.append({"cycle": cyc, "crossover_dte": sub["dte"].max()})
        xdf = pd.DataFrame(rows)
        if not xdf.empty:
            self.log.info(
                f"Liquidity crossover — cycles: {len(xdf)} | "
                f"median DTE: {xdf['crossover_dte'].median():.1f} | "
                f"mean DTE: {xdf['crossover_dte'].mean():.1f}"
            )
        return xdf

    def aggregate_volume_share(self) -> pd.DataFrame:
        """Aggregate back-month volume share by DTE across all cycles."""
        agg = self.panel.groupby("dte").apply(
            lambda g: g["next_vol"].sum() / (g["near_vol"].sum() + g["next_vol"].sum())
        ).reset_index()
        agg.columns = ["dte", "back_share"]
        return agg


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — STATISTICAL ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class StatisticalAnalyzer:
    """
    Rigorous statistical tests:
      - Kruskal-Wallis (are DTE-window distributions different?)
      - Pairwise Mann-Whitney U (which windows differ?)
      - AR(1) persistence of the spread series
      - Bootstrap 95 % CI of mean spread per window
    """

    STRATEGY_WINDOWS = {
        "T-0~2 (Settlement)": (0, 2),
        "T-3~5":              (3, 5),
        "T-6~8":              (6, 8),
        "T-9~12":             (9, 12),
        "T-13~17":            (13, 17),
        "T-18~25":            (18, 25),
        "T-26~40 (Early)":    (26, 40),
    }

    def __init__(self, panel: pd.DataFrame, logger: logging.Logger):
        self.panel = panel.copy()
        self.log = logger
        self._sim: Optional[pd.DataFrame] = None

    # ── per-cycle simulation table ────────────────────────────────────────────

    def build_simulation(self) -> pd.DataFrame:
        """For each cycle × DTE window: record median spread as realized roll cost."""
        rows = []
        cycles = self.panel["near_month"].unique()
        for cyc in cycles:
            sub = self.panel[self.panel["near_month"] == cyc]
            for strat, (lo, hi) in self.STRATEGY_WINDOWS.items():
                win = sub[(sub["dte"] >= lo) & (sub["dte"] <= hi)]
                if win.empty:
                    continue
                sp = win["spread"].median()
                rows.append({
                    "cycle":      cyc,
                    "strategy":   strat,
                    "dte_lo":     lo,
                    "dte_hi":     hi,
                    "spread":     sp,
                    "long_cost":  sp,
                    "short_cost": -sp,
                    "n_obs":      len(win),
                })
        self._sim = pd.DataFrame(rows)
        return self._sim

    def strategy_aggregate(self) -> pd.DataFrame:
        """Aggregate simulation by strategy."""
        if self._sim is None:
            self.build_simulation()
        sim = self._sim
        strat_order = list(self.STRATEGY_WINDOWS.keys())
        agg = sim.groupby("strategy").agg(
            mean_spread    = ("spread",     "mean"),
            median_spread  = ("spread",     "median"),
            std_spread     = ("spread",     "std"),
            mean_long_cost = ("long_cost",  "mean"),
            mean_short_cost= ("short_cost", "mean"),
            n_cycles       = ("cycle",      "count"),
            pct_backw      = ("spread",     lambda x: (x>0).mean()*100),
            pct_contango   = ("spread",     lambda x: (x<0).mean()*100),
        ).reset_index()
        agg["_ord"] = agg["strategy"].map({s: i for i, s in enumerate(strat_order)})
        agg = agg.sort_values("_ord").drop("_ord", axis=1).reset_index(drop=True)
        return agg

    def kruskal_wallis(self) -> Tuple[float, float]:
        if self._sim is None:
            self.build_simulation()
        groups = {
            s: self._sim[self._sim["strategy"] == s]["spread"].dropna().values
            for s in self.STRATEGY_WINDOWS
        }
        groups_clean = {k: v for k, v in groups.items() if len(v) >= 3}
        stat, p = kruskal(*groups_clean.values())
        self.log.info(
            f"Kruskal-Wallis H={stat:.3f}, p={p:.4f} → "
            f"{'SIGNIFICANT' if p<0.05 else 'not significant'} at α=0.05"
        )
        return stat, p

    def pairwise_mannwhitney(self) -> Dict[Tuple[str,str], float]:
        if self._sim is None:
            self.build_simulation()
        groups = {
            s: self._sim[self._sim["strategy"] == s]["spread"].dropna().values
            for s in self.STRATEGY_WINDOWS
        }
        groups_clean = {k: v for k, v in groups.items() if len(v) >= 3}
        strat_list = list(groups_clean.keys())
        pairwise = {}
        for i, a in enumerate(strat_list):
            for j, b in enumerate(strat_list):
                if i >= j:
                    continue
                _, p = mannwhitneyu(groups_clean[a], groups_clean[b], alternative="two-sided")
                pairwise[(a, b)] = p
        return pairwise

    def ar1_persistence(self) -> Dict[str, float]:
        """AR(1) regression on spread across all cycles."""
        ps = self.panel.sort_values("date").copy()
        ps["spread_lag1"] = ps.groupby("near_month")["spread"].shift(1)
        valid = ps.dropna(subset=["spread", "spread_lag1"])
        slope, intercept, r, p, se = stats.linregress(valid["spread_lag1"], valid["spread"])
        thresh = valid["spread_lag1"].median()
        pred = np.where(valid["spread_lag1"] > thresh, "positive", "negative")
        actual = np.where(valid["spread"] > 0, "positive", "negative")
        dir_acc = (pred == actual).mean()
        self.log.info(
            f"AR(1) β={slope:.4f}, R²={r**2:.4f}, p={p:.2e} | "
            f"directional accuracy: {dir_acc:.1%}"
        )
        return {
            "beta": slope, "intercept": intercept,
            "r_squared": r**2, "p_value": p,
            "directional_accuracy": dir_acc,
        }

    def bootstrap_ci(self, n_boot: int = 2000) -> pd.DataFrame:
        """95 % bootstrap CI of mean spread per DTE window."""
        if self._sim is None:
            self.build_simulation()
        rows = []
        for s in self.STRATEGY_WINDOWS:
            vals = self._sim[self._sim["strategy"] == s]["spread"].dropna().values
            if len(vals) < 5:
                rows.append({"strategy": s, "ci_lo": np.nan, "ci_hi": np.nan, "mean": np.nan})
                continue
            boot = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
            rows.append({
                "strategy": s,
                "ci_lo": float(np.percentile(boot, 2.5)),
                "ci_hi": float(np.percentile(boot, 97.5)),
                "mean":  float(np.mean(vals)),
            })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — REGIME ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class RegimeAnalyzer:
    """
    Classify each roll cycle by its price trend:
      Bull   : cycle return > +2 %
      Bear   : cycle return < −2 %
      Neutral: ±2 %
    Then compute per-regime spread by DTE window.
    """

    DTE_LABELS = ["T-0~2","T-3~5","T-6~8","T-9~12","T-13~17","T-18~25","T-26~40"]

    def __init__(self, panel: pd.DataFrame, logger: logging.Logger):
        self.panel = panel.copy()
        self.log = logger

    def classify(self) -> pd.DataFrame:
        rows = []
        for cyc, grp in self.panel.groupby("near_month"):
            sub = grp.sort_values("dte", ascending=False)
            if len(sub) < 5:
                continue
            ret = (sub.iloc[-1]["near_close"] - sub.iloc[0]["near_close"]) / sub.iloc[0]["near_close"] * 100
            regime = "Bull" if ret > 2 else ("Bear" if ret < -2 else "Neutral")
            for dbin, spval in sub.groupby("dte_bin")["spread"].median().items():
                rows.append({"cycle": cyc, "regime": regime, "dte_bin": dbin, "spread": spval})

        reg_df = pd.DataFrame(rows)
        counts = reg_df.groupby("cycle")["regime"].first().value_counts().to_dict()
        self.log.info(f"Market regime breakdown: {counts}")
        return reg_df, counts


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — INTRADAY ANALYZER 
# ══════════════════════════════════════════════════════════════════════════════

class IntradayAnalyzer:
    """
    Analyze the best intraday execution window for rolling.

    Bug fixed: the original roll_deep_analysis.py tried to access 'BTime'
    on sprd1m which was a daily-aggregated DataFrame (BTime was grouped away).
    We now receive sprd_1m_raw — the raw minute-bar 1-month spread contracts
    that still have the BTime column.
    """

    REGULAR_SESSION_HOURS = (8, 13)   # 08:xx – 13:xx inclusive

    def __init__(self, sprd_1m_raw: pd.DataFrame, outr: pd.DataFrame, logger: logging.Logger):
        self.sprd_raw = sprd_1m_raw.copy()
        self.outr = outr.copy()
        self.log = logger

    def _parse_hour(self, df: pd.DataFrame, col: str = "BTime") -> pd.DataFrame:
        """Extract integer hour from BTime string (HHMM or HH:MM)."""
        if col not in df.columns:
            self.log.warning(f"Column '{col}' not found — skipping intraday parse")
            return df
        df = df.copy()
        # Handle both "HHMM" and "HH:MM" formats
        btime = df[col].astype(str).str.replace(":", "", regex=False)
        df["hour"] = btime.str[:2].astype(int, errors="ignore")
        return df

    def hourly_spread_contract(self) -> pd.DataFrame:
        """Volume and price of roll-spread contracts aggregated by hour."""
        if self.sprd_raw.empty or "BTime" not in self.sprd_raw.columns:
            self.log.warning("No raw spread data with BTime for intraday analysis")
            return pd.DataFrame()
        df = self._parse_hour(self.sprd_raw)
        lo, hi = self.REGULAR_SESSION_HOURS
        df = df[(df["hour"] >= lo) & (df["hour"] <= hi)]
        agg = df.groupby("hour").agg(
            mean_spread_px = ("Px_C",      "mean"),
            median_spread_px=("Px_C",      "median"),
            total_vol      = ("Trade_Vol", "sum"),
            mean_vol       = ("Trade_Vol", "mean"),
            count          = ("Px_C",      "count"),
        ).reset_index()
        self.log.info(f"Intraday spread contract analysis: {len(agg)} hourly buckets")
        return agg

    def hourly_outright_spread(self) -> pd.DataFrame:
        """Near − next spread computed from outright minute bars, by hour."""
        if self.outr.empty or "BTime" not in self.outr.columns:
            return pd.DataFrame()
        outr = self._parse_hour(self.outr)
        lo, hi = self.REGULAR_SESSION_HOURS
        outr = outr[(outr["hour"] >= lo) & (outr["hour"] <= hi) & (outr["days_to_expiry"] >= 0)]

        # Rank DTE within each (date, BTime): 1=near, 2=next
        outr["rank_dte"] = outr.groupby(["Txdate", "BTime"])["days_to_expiry"].rank(method="first")
        near = outr[outr["rank_dte"] == 1][["Txdate","BTime","hour","Px_C","Trade_Vol","days_to_expiry"]].copy()
        nxt  = outr[outr["rank_dte"] == 2][["Txdate","BTime","Px_C","Trade_Vol"]].copy()
        near = near.rename(columns={"Px_C":"near_px","Trade_Vol":"near_vol","days_to_expiry":"near_dte"})
        nxt  = nxt.rename(columns={"Px_C":"next_px","Trade_Vol":"next_vol"})

        merged = near.merge(nxt, on=["Txdate","BTime"], how="inner")
        merged["spread_intra"]   = merged["near_px"] - merged["next_px"]
        merged["vol_ratio_intra"]= merged["next_vol"] / merged["near_vol"].replace(0, np.nan)

        agg = merged.groupby("hour").agg(
            mean_spread    = ("spread_intra",    "mean"),
            median_spread  = ("spread_intra",    "median"),
            mean_vol_ratio = ("vol_ratio_intra", "mean"),
        ).reset_index()
        return agg


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — ROLL TIMING ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class RollTimingAnalyzer:
    """
    Composite roll-cost + liquidity ranking to identify optimal DTE windows.

    Long rollers:  sell near, buy back → prefer LOW spread (small roll cost)
    Short rollers: buy near, sell back → prefer HIGH spread (collect roll income)
    Score = 50 % cost (inverted) + 50 % liquidity (log-volume)
    """

    def __init__(self, strat_agg: pd.DataFrame, logger: logging.Logger):
        self.agg = strat_agg.copy()
        self.log = logger

    def rank(self) -> pd.DataFrame:
        df = self.agg.copy()
        for col in ["mean_long_cost", "mean_short_cost"]:
            rng = df[col].max() - df[col].min()
            df[f"{col}_norm"] = (df[col] - df[col].min()) / rng if rng > 0 else 0.5

        # Composite: high short_cost_norm = high spread = good for long;
        # low long_cost_norm = low spread = good for short (inverted)
        df["long_score"]  = 1 - df["mean_long_cost_norm"]   # lower cost = higher score
        df["short_score"] = df["mean_short_cost_norm"]       # higher spread = higher score
        df["long_rank"]   = df["long_score"].rank(ascending=False).astype(int)
        df["short_rank"]  = df["short_score"].rank(ascending=False).astype(int)

        best_long  = df.loc[df["long_score"].idxmax(),  "strategy"]
        best_short = df.loc[df["short_score"].idxmax(), "strategy"]
        self.log.info(f"Best roll window — LONG:  {best_long}")
        self.log.info(f"Best roll window — SHORT: {best_short}")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════

class RollVisualizer:
    """Generate all analysis charts and save to PNG files."""

    C = dict(
        long="#2166AC", short="#B2182B", spread="#7570B3",
        front="#1B9E77", back="#D95F02", highlight="#E7298A",
        bull="#33A02C", bear="#E31A1C", neutral="#888888", band="#BBBBBB",
    )

    def __init__(self, output_dir: str, logger: logging.Logger):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = logger

    def _save(self, fig, name: str):
        p = self.out / f"{name}.png"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.log.info(f"Chart saved → {p}")

    @staticmethod
    def _fmt_k(x, _): return f"{x/1000:.0f}K" if abs(x) >= 1000 else f"{x:.0f}"

    # ── Chart 1: All-cycle spread trajectories ────────────────────────────────

    def plot_cycle_trajectories(self, panel: pd.DataFrame):
        cycles = sorted(panel["near_month"].unique())
        all_dte = panel.groupby("dte")["spread"].agg(
            ["median","mean",
             lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        )
        all_dte.columns = ["median","mean","q25","q75"]

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        ax = axes[0]
        for cyc in cycles:
            ser = panel[panel["near_month"] == cyc].set_index("dte")["spread"]
            if len(ser) < 5:
                continue
            col = (self.C["bull"] if ser.mean() > 10
                   else self.C["bear"] if ser.mean() < -10
                   else self.C["neutral"])
            ax.plot(ser.index, ser.values, alpha=0.4, linewidth=0.8, color=col)
        ax.plot(all_dte.index, all_dte["median"], color="black", linewidth=2.2,
                label="Median across cycles", zorder=5)
        ax.fill_between(all_dte.index, all_dte["q25"], all_dte["q75"],
                        alpha=0.18, color="black", label="IQR")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.invert_xaxis()
        ax.set_xlabel("Days to Expiry (DTE)")
        ax.set_ylabel("Spread (Front − Back, pts)")
        ax.set_title("(a) Per-Cycle Near-Month Spread Trajectory  "
                     "[Green=avg backwardation | Red=avg contango]")
        ax.legend(loc="upper left")

        ax = axes[1]
        for cyc in cycles:
            sub_v = panel[panel["near_month"] == cyc].set_index("dte")["vol_ratio"]
            ax.plot(sub_v.index, sub_v.values, alpha=0.3, linewidth=0.7, color=self.C["spread"])
        vol_med = panel.groupby("dte")["vol_ratio"].median()
        ax.plot(vol_med.index, vol_med.values, color=self.C["back"], linewidth=2.2,
                label="Median vol ratio", zorder=5)
        ax.axhline(1.0, color=self.C["bear"], linewidth=1.5, linestyle="--", label="Parity 1:1")
        ax.invert_xaxis()
        ax.set_xlabel("Days to Expiry (DTE)")
        ax.set_ylabel("Back / Front Volume Ratio")
        ax.set_title("(b) Liquidity Migration — Back-Month Volume / Near-Month Volume")
        ax.legend(loc="upper right")

        fig.suptitle("FITX Near-Month Roll: All Cycles — Spread & Volume Trajectories",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        self._save(fig, "01_cycle_trajectories")

    # ── Chart 2: Liquidity crossover ─────────────────────────────────────────

    def plot_liquidity_crossover(self, xdf: pd.DataFrame, vol_share: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.hist(xdf["crossover_dte"], bins=15, color=self.C["spread"], alpha=0.7, edgecolor="white")
        ax.axvline(xdf["crossover_dte"].median(), color=self.C["highlight"], linewidth=2,
                   linestyle="--", label=f"Median={xdf['crossover_dte'].median():.0f} DTE")
        ax.axvline(xdf["crossover_dte"].mean(), color=self.C["back"], linewidth=2,
                   linestyle=":", label=f"Mean={xdf['crossover_dte'].mean():.1f} DTE")
        ax.set_xlabel("DTE at Liquidity Crossover")
        ax.set_ylabel("# Cycles")
        ax.set_title("(a) Distribution of Liquidity Crossover DTE\n(back-month vol ≥ front-month vol)")
        ax.legend()

        ax = axes[1]
        ax.fill_between(vol_share["dte"], vol_share["back_share"],
                        alpha=0.4, color=self.C["back"], label="Back-month share")
        ax.fill_between(vol_share["dte"], vol_share["back_share"], 1,
                        alpha=0.3, color=self.C["front"], label="Near-month share")
        ax.axhline(0.5, color="black", linewidth=1, linestyle="--", label="50% parity")
        ax.axvline(xdf["crossover_dte"].median(), color=self.C["highlight"], linewidth=1.5,
                   linestyle=":", label=f"Median crossover DTE={xdf['crossover_dte'].median():.0f}")
        ax.invert_xaxis()
        ax.set_xlabel("Days to Expiry")
        ax.set_ylabel("Volume Share")
        ax.set_title("(b) Aggregate Volume Share: Near vs Back Month by DTE")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(loc="upper right")

        fig.suptitle("Liquidity Migration Analysis", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save(fig, "02_liquidity_crossover")

    # ── Chart 3: Roll cost simulation ─────────────────────────────────────────

    def plot_roll_cost(self, strat_agg: pd.DataFrame, sim: pd.DataFrame):
        strat_order = list(strat_agg["strategy"])
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # (a) Long cost
        ax = fig.add_subplot(gs[0, 0])
        colors_l = [self.C["bull"] if v < 0 else self.C["bear"] for v in strat_agg["mean_long_cost"]]
        bars = ax.barh(strat_agg["strategy"], strat_agg["mean_long_cost"],
                       color=colors_l, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1)
        for bar, val in zip(bars, strat_agg["mean_long_cost"]):
            ax.text(val + (1 if val >= 0 else -1),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.1f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8)
        ax.set_title("(a) Long Roll Cost by DTE Window\n(−=favourable / +=cost)")
        ax.set_xlabel("Avg Spread (pts)")
        ax.invert_yaxis()

        # (b) Short benefit
        ax = fig.add_subplot(gs[0, 1])
        colors_s = [self.C["bull"] if v > 0 else self.C["bear"] for v in strat_agg["mean_short_cost"]]
        bars2 = ax.barh(strat_agg["strategy"], strat_agg["mean_short_cost"],
                        color=colors_s, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1)
        for bar, val in zip(bars2, strat_agg["mean_short_cost"]):
            ax.text(val + (1 if val >= 0 else -1),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.1f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8)
        ax.set_title("(b) Short Roll Benefit by DTE Window\n(+=favourable / −=cost)")
        ax.set_xlabel("Avg −Spread (pts)")
        ax.invert_yaxis()

        # (c) % backwardation vs contango
        ax = fig.add_subplot(gs[0, 2])
        y_pos = range(len(strat_agg))
        ax.barh(y_pos, strat_agg["pct_backw"], color=self.C["long"],
                alpha=0.7, label="Backwardation %", height=0.4, align="center")
        ax.barh([y + 0.45 for y in y_pos], strat_agg["pct_contango"], color=self.C["short"],
                alpha=0.7, label="Contango %", height=0.4, align="center")
        ax.set_yticks([y + 0.225 for y in y_pos])
        ax.set_yticklabels(strat_agg["strategy"], fontsize=8)
        ax.axvline(50, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title("(c) % Cycles in Backwardation vs Contango")
        ax.set_xlabel("% of Cycles")
        ax.legend(loc="lower right", fontsize=8)
        ax.invert_yaxis()

        # (d) Box plot
        ax = fig.add_subplot(gs[1, :2])
        box_data = [sim[sim["strategy"] == s]["spread"].dropna().values for s in strat_order]
        bp = ax.boxplot(box_data, vert=True, patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.3),
                        medianprops=dict(color="black", linewidth=1.5), widths=0.5)
        for patch in bp["boxes"]:
            patch.set_facecolor(self.C["spread"])
            patch.set_alpha(0.5)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(range(1, len(strat_order) + 1))
        ax.set_xticklabels([s.split(" ")[0] for s in strat_order], rotation=20, fontsize=9)
        ax.set_ylabel("Spread (pts)")
        ax.set_title("(d) Distribution of Realized Roll Spread per DTE Window (IQR box)")

        # (e) Table
        ax = fig.add_subplot(gs[1, 2])
        ax.axis("off")
        tbl = ax.table(
            cellText=[[r["strategy"].split("(")[0].strip(),
                       f"{r['n_cycles']:.0f}",
                       f"{r['mean_long_cost']:+.1f}",
                       f"{r['mean_short_cost']:+.1f}"]
                      for _, r in strat_agg.iterrows()],
            colLabels=["DTE Window","N","LongCost","ShortBen"],
            cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
        )
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        for j in range(4):
            tbl[0, j].set_facecolor("#DDDDDD")
        for i, (_, row) in enumerate(strat_agg.iterrows(), 1):
            tbl[i, 2].set_facecolor("#C6EFC8" if row["mean_long_cost"] < 0 else "#F5C6C6")
            tbl[i, 3].set_facecolor("#C6EFC8" if row["mean_short_cost"] > 0 else "#F5C6C6")
        ax.set_title("(e) Summary Table", pad=10, fontsize=10)

        fig.suptitle("Realized Roll Cost Simulation by DTE Strategy",
                     fontsize=14, fontweight="bold")
        self._save(fig, "03_roll_cost_simulation")

    # ── Chart 4: Statistical tests ────────────────────────────────────────────

    def plot_statistical_tests(self, kw_stat: float, kw_p: float,
                               pairwise: dict, ci_df: pd.DataFrame,
                               strat_order: List[str]):
        short_labels = [s.split("(")[0].strip().replace("T-","") for s in strat_order]
        n = len(strat_order)

        # Build p-matrix
        pmat = np.full((n, n), np.nan)
        idx_map = {s: i for i, s in enumerate(strat_order)}
        for (a, b), p in pairwise.items():
            i, j = idx_map.get(a, -1), idx_map.get(b, -1)
            if i >= 0 and j >= 0:
                pmat[i, j] = p
                pmat[j, i] = p

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        im = ax.imshow(pmat, cmap="RdYlGn_r", vmin=0, vmax=0.3, aspect="auto")
        plt.colorbar(im, ax=ax, label="p-value")
        ax.set_xticks(range(n)); ax.set_xticklabels(short_labels, rotation=30, fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(short_labels, fontsize=8)
        for i in range(n):
            for j in range(n):
                if not np.isnan(pmat[i, j]):
                    color = "white" if pmat[i, j] < 0.1 else "black"
                    ax.text(j, i, f"{pmat[i,j]:.2f}", ha="center", va="center",
                            fontsize=7, color=color)
        ax.set_title(f"(a) Pairwise Mann-Whitney p-values\n"
                     f"KW: H={kw_stat:.1f}, p={kw_p:.3f} | green=significant")

        ax = axes[1]
        y = range(len(strat_order))
        ax.scatter(ci_df["mean"], list(y), color=self.C["spread"], s=60, zorder=5)
        for i, row in ci_df.iterrows():
            if not np.isnan(row["ci_lo"]):
                ax.plot([row["ci_lo"], row["ci_hi"]], [i, i],
                        color=self.C["spread"], linewidth=2.5, alpha=0.7)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(list(y))
        ax.set_yticklabels([s.split("(")[0].strip() for s in strat_order], fontsize=9)
        ax.set_xlabel("Mean Spread (pts)")
        ax.set_title("(b) 95% Bootstrap CI of Mean Spread per DTE Window\n"
                     "CI crossing zero → no significant directional bias")
        ax.invert_yaxis()

        fig.suptitle("Statistical Significance of Roll Timing Differences",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "04_statistical_tests")

    # ── Chart 5: Market regime ────────────────────────────────────────────────

    def plot_regime(self, reg_df: pd.DataFrame):
        labels = ["T-0~2","T-3~5","T-6~8","T-9~12","T-13~17","T-18~25","T-26~40"]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for ax, regime, col in zip(axes, ["Bull","Neutral","Bear"],
                                    [self.C["bull"], self.C["neutral"], self.C["bear"]]):
            sub = reg_df[reg_df["regime"] == regime]
            agg = sub.groupby("dte_bin")["spread"].agg(
                ["mean","median",
                 lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
            )
            agg.columns = ["mean","median","q25","q75"]
            agg = agg.reindex(labels)
            x = range(len(agg))
            ax.fill_between(x, agg["q25"], agg["q75"], alpha=0.25, color=col)
            ax.plot(x, agg["mean"],   color=col, linewidth=2, label="Mean")
            ax.plot(x, agg["median"], color=col, linewidth=1.5, linestyle="--", label="Median")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=30, fontsize=8)
            n_cyc = sub["cycle"].nunique()
            ax.set_title(f"{regime} Market ({n_cyc} cycles)")
            ax.set_xlabel("DTE Window")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Roll Spread (pts)")
        fig.suptitle("Roll Spread by DTE Window — Market Regime Conditioning\n"
                     "(Bull: cycle return>2% | Bear: <−2% | Neutral: ±2%)",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save(fig, "05_regime_analysis")

    # ── Chart 6: Spread autocorrelation ──────────────────────────────────────

    def plot_autocorrelation(self, panel: pd.DataFrame, ar1: dict):
        ps = panel.sort_values("date").copy()
        ps["spread_lag1"] = ps.groupby("near_month")["spread"].shift(1)
        valid = ps.dropna(subset=["spread","spread_lag1"])
        slope, intercept = ar1["beta"], ar1["intercept"]
        r2 = ar1["r_squared"]

        acf_vals = []
        for lag in range(1, 11):
            lagged = ps.groupby("near_month")["spread"].shift(lag)
            acf_vals.append(ps["spread"].corr(lagged))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        ax = axes[0]
        ax.scatter(valid["spread_lag1"], valid["spread"], alpha=0.15, s=5, color=self.C["spread"])
        xfit = np.linspace(valid["spread_lag1"].min(), valid["spread_lag1"].max(), 100)
        ax.plot(xfit, slope * xfit + intercept, color=self.C["highlight"], linewidth=2,
                label=f"AR(1): β={slope:.3f}, R²={r2:.3f}")
        ax.axhline(0, color="black", linewidth=0.5); ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Spread (t−1) pts"); ax.set_ylabel("Spread (t) pts")
        ax.set_title("(a) AR(1): Today vs Yesterday's Spread")
        ax.legend()

        ax = axes[1]
        conf = 1.96 / np.sqrt(len(valid))
        ax.bar(range(1, 11), acf_vals, color=self.C["spread"], alpha=0.7, edgecolor="white")
        ax.axhline( conf, color="red", linewidth=1, linestyle="--", label=f"95% CI ±{conf:.3f}")
        ax.axhline(-conf, color="red", linewidth=1, linestyle="--")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Lag (days)"); ax.set_ylabel("Autocorrelation")
        ax.set_title("(b) Spread ACF (lags 1–10)")
        ax.legend()

        ax = axes[2]
        window = 60
        rolling_r2, rolling_dates = [], []
        for i in range(window, len(ps)):
            win = ps.iloc[i-window:i].dropna(subset=["spread","spread_lag1"])
            if len(win) < 20:
                continue
            try:
                _, _, r_w, _, _ = stats.linregress(win["spread_lag1"], win["spread"])
                rolling_r2.append(r_w**2)
                rolling_dates.append(ps.iloc[i]["date"])
            except Exception:
                pass
        ax.plot(rolling_dates, rolling_r2, color=self.C["spread"], linewidth=1.2)
        ax.fill_between(rolling_dates, rolling_r2, alpha=0.3, color=self.C["spread"])
        ax.axhline(r2, color=self.C["highlight"], linewidth=1.5, linestyle="--",
                   label=f"Overall R²={r2:.3f}")
        ax.set_ylabel("Rolling 60-day AR(1) R²")
        ax.set_title("(c) Rolling AR(1) R² — Spread Persistence Over Time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.legend()

        fig.suptitle("Spread Autocorrelation & Predictability Analysis",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save(fig, "06_spread_autocorrelation")

    # ── Chart 7: Intraday execution ───────────────────────────────────────────

    def plot_intraday(self, hourly_sc: pd.DataFrame, hourly_outr: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        ax = axes[0, 0]
        if not hourly_sc.empty and "total_vol" in hourly_sc.columns:
            ax.bar(hourly_sc["hour"], hourly_sc["total_vol"], color=self.C["back"], alpha=0.7)
        ax.set_title("(a) Roll-Spread Contract: Total Volume by Hour")
        ax.set_xlabel("Hour"); ax.set_ylabel("Total Contracts")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._fmt_k))

        ax = axes[0, 1]
        if not hourly_sc.empty and "mean_spread_px" in hourly_sc.columns:
            ax.plot(hourly_sc["hour"], hourly_sc["mean_spread_px"],
                    color=self.C["spread"], marker="o", linewidth=2)
            ax.fill_between(hourly_sc["hour"],
                            hourly_sc["median_spread_px"], hourly_sc["mean_spread_px"],
                            alpha=0.3, color=self.C["spread"])
        ax.set_title("(b) Roll-Spread Contract: Avg Price by Hour")
        ax.set_xlabel("Hour"); ax.set_ylabel("Spread Price (pts)")

        ax = axes[1, 0]
        if not hourly_outr.empty:
            ax.plot(hourly_outr["hour"], hourly_outr["mean_spread"],
                    color=self.C["long"], marker="o", linewidth=2, label="Mean")
            ax.plot(hourly_outr["hour"], hourly_outr["median_spread"],
                    color=self.C["long"], marker="s", linewidth=1.5, linestyle="--", label="Median")
            ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("(c) Outright-Implied Spread by Hour")
        ax.set_xlabel("Hour"); ax.set_ylabel("Near − Next (pts)")
        ax.legend()

        ax = axes[1, 1]
        if not hourly_outr.empty and "mean_vol_ratio" in hourly_outr.columns:
            ax.plot(hourly_outr["hour"], hourly_outr["mean_vol_ratio"],
                    color=self.C["short"], marker="o", linewidth=2)
            ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="Parity")
        ax.set_title("(d) Back/Front Volume Ratio by Hour")
        ax.set_xlabel("Hour"); ax.set_ylabel("Back / Front Vol Ratio")
        ax.legend()

        fig.suptitle("Intraday Execution Analysis — Best Time to Roll",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        self._save(fig, "07_intraday_execution")

    # ── Chart 8: YoY evolution + tail risk ───────────────────────────────────

    def plot_yoy_tail(self, panel: pd.DataFrame, sim: pd.DataFrame, strat_order: List[str]):
        labels = ["T-0~2","T-3~5","T-6~8","T-9~12","T-13~17","T-18~25","T-26~40"]
        panel = panel.copy()
        panel["year"] = panel["date"].dt.year
        yearly = panel.groupby(["year","dte_bin"])["spread"].agg(["mean","std","median"]).reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for yr, col in zip([2023, 2024, 2025, 2026],
                            [self.C["neutral"], self.C["bull"], self.C["long"], self.C["bear"]]):
            sub = yearly[yearly["year"] == yr]
            if sub.empty: continue
            ax.plot(range(len(sub)), sub["mean"], color=col, linewidth=2,
                    marker="o", markersize=5, label=str(yr))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=25, fontsize=8)
        ax.set_ylabel("Mean Spread (pts)")
        ax.set_title("(a) Mean Roll Spread by DTE Window — Year-over-Year")
        ax.legend()

        ax = axes[1]
        tail_95 = sim.groupby("strategy")["spread"].quantile(0.95).reindex(strat_order)
        tail_05 = sim.groupby("strategy")["spread"].quantile(0.05).reindex(strat_order)
        tail_mn = sim.groupby("strategy")["spread"].mean().reindex(strat_order)
        x = range(len(strat_order))
        sl2 = [s.split("(")[0].strip() for s in strat_order]
        ax.fill_between(x, tail_05.values, tail_95.values, alpha=0.2, color=self.C["spread"],
                        label="5th–95th pctile")
        ax.plot(x, tail_mn.values, color=self.C["spread"], linewidth=2, marker="o", label="Mean")
        ax.plot(x, tail_95.values, color=self.C["bear"], linewidth=1.5, linestyle="--",
                label="95th pctile (worst long cost)")
        ax.plot(x, tail_05.values, color=self.C["bull"], linewidth=1.5, linestyle="--",
                label="5th pctile (best long opportunity)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(list(x)); ax.set_xticklabels(sl2, rotation=25, fontsize=8)
        ax.set_ylabel("Spread (pts)")
        ax.set_title("(b) Tail Risk — Best/Worst Realized Spread per DTE Window")
        ax.legend(fontsize=8)

        fig.suptitle("Year-over-Year Evolution & Tail Risk",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save(fig, "08_yoy_tail_risk")

    # ── Chart 9: Decision matrix ──────────────────────────────────────────────

    def plot_decision_matrix(self, ranked: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        x = np.arange(len(ranked))
        w = 0.35
        ax = axes[0]
        ax.bar(x - w/2, ranked["long_score"],  w, label="Long score",  color=self.C["long"],  alpha=0.8)
        ax.bar(x + w/2, ranked["short_score"], w, label="Short score", color=self.C["short"], alpha=0.8)
        best_l = ranked["long_score"].idxmax()
        best_s = ranked["short_score"].idxmax()
        ax.annotate("★ LONG", (x[best_l] - w/2, ranked.loc[best_l,"long_score"]),
                    textcoords="offset points", xytext=(0,6), ha="center",
                    fontsize=9, fontweight="bold", color=self.C["long"])
        ax.annotate("★ SHORT", (x[best_s] + w/2, ranked.loc[best_s,"short_score"]),
                    textcoords="offset points", xytext=(0,6), ha="center",
                    fontsize=9, fontweight="bold", color=self.C["short"])
        ax.set_xticks(x)
        ax.set_xticklabels(ranked["strategy"].str.split("(").str[0].str.strip(),
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Score (higher = better)")
        ax.set_title("(a) Composite Roll Score: Long vs Short")
        ax.legend()

        ax = axes[1]
        ax.axis("off")
        tbl = ax.table(
            cellText=[[r["strategy"].split("(")[0].strip(),
                       f"{r['mean_long_cost']:+.1f}",
                       f"{r['mean_short_cost']:+.1f}",
                       f"{r['pct_backw']:.0f}%",
                       f"L:{r['long_rank']} S:{r['short_rank']}"]
                      for _, r in ranked.iterrows()],
            colLabels=["Window","LongCost","ShortBen","%Backw","Rank"],
            cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
        )
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        for j in range(5):
            tbl[0, j].set_facecolor("#CCCCCC")

        fig.suptitle("Roll Timing Decision Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "09_decision_matrix")



# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class FITXRollPipeline:
    """
    End-to-end pipeline orchestrator.

    Usage:
        pipeline = FITXRollPipeline("data.csv")
        pipeline.run()
    """

    def __init__(self, filepath: str,
                 output_dir: str = "output_combined",
                 findings_dir: str = "findings"):
        self.filepath     = filepath
        self.output_dir   = output_dir
        self.findings_dir = findings_dir
        self.log = setup_logging()
        configure_plot_style()

    def run(self) -> dict:
        self.log.info("=" * 70)
        self.log.info("FITX ROLL COMBINED PIPELINE — START")
        self.log.info("=" * 70)

        # ── Stage 1: Load & Clean ─────────────────────────────────────────────
        self.log.info("[1/8] Loading & cleaning data...")
        loader = FITXDataLoader(self.filepath, self.log)
        df_raw = loader.load()
        df     = loader.clean(df_raw)

        # ── Stage 2: Build Roll Panel ─────────────────────────────────────────
        self.log.info("[2/8] Building roll panel...")
        builder       = RollPanelBuilder(df, self.log)
        panel, sprd_1m = builder.build()

        # ── Stage 3: Roll Yield ───────────────────────────────────────────────
        self.log.info("[3/8] Computing roll yield...")
        ry_analyzer = RollYieldAnalyzer(panel, self.log)
        panel_ry    = ry_analyzer.compute()   # panel enriched with roll_yield_pct

        # ── Stage 4: Liquidity ────────────────────────────────────────────────
        self.log.info("[4/8] Analyzing liquidity...")
        lq_analyzer = LiquidityAnalyzer(panel_ry, self.log)
        xdf         = lq_analyzer.crossover_dtes()
        vol_share   = lq_analyzer.aggregate_volume_share()

        # ── Stage 5: Statistics ───────────────────────────────────────────────
        self.log.info("[5/8] Running statistical tests...")
        stat_analyzer = StatisticalAnalyzer(panel_ry, self.log)
        sim           = stat_analyzer.build_simulation()
        strat_agg     = stat_analyzer.strategy_aggregate()
        kw_stat, kw_p = stat_analyzer.kruskal_wallis()
        pairwise      = stat_analyzer.pairwise_mannwhitney()
        ar1           = stat_analyzer.ar1_persistence()
        ci_df         = stat_analyzer.bootstrap_ci()
        strat_order   = list(StatisticalAnalyzer.STRATEGY_WINDOWS.keys())

        # ── Stage 5b: Regime ──────────────────────────────────────────────────
        self.log.info("[5b/8] Regime conditioning...")
        regime_analyzer = RegimeAnalyzer(panel_ry, self.log)
        reg_df, regime_counts = regime_analyzer.classify()

        # ── Stage 6: Intraday ─────────────────────────────────────────────────
        self.log.info("[6/8] Intraday execution analysis...")
        outr_raw = df[~df["is_roll_spread"]].copy()
        intraday = IntradayAnalyzer(sprd_1m, outr_raw, self.log)
        hourly_sc   = intraday.hourly_spread_contract()
        hourly_outr = intraday.hourly_outright_spread()

        # ── Stage 7: Roll Timing ──────────────────────────────────────────────
        self.log.info("[7/8] Ranking roll timing windows...")
        timing_analyzer = RollTimingAnalyzer(strat_agg, self.log)
        ranked = timing_analyzer.rank()

        best_long_row  = ranked.loc[ranked["long_score"].idxmax()]
        best_short_row = ranked.loc[ranked["short_score"].idxmax()]

        # ── Stage 8: Visualize ────────────────────────────────────────────────
        self.log.info("[8/8] Generating charts...")
        viz = RollVisualizer(self.output_dir, self.log)
        viz.plot_cycle_trajectories(panel_ry)
        viz.plot_liquidity_crossover(xdf, vol_share)
        viz.plot_roll_cost(strat_agg, sim)
        viz.plot_statistical_tests(kw_stat, kw_p, pairwise, ci_df, strat_order)
        viz.plot_regime(reg_df)
        viz.plot_autocorrelation(panel_ry, ar1)
        viz.plot_intraday(hourly_sc, hourly_outr)
        viz.plot_yoy_tail(panel_ry, sim, strat_order)
        viz.plot_decision_matrix(ranked)


        self.log.info("=" * 70)
        self.log.info("FITX ROLL COMBINED PIPELINE — COMPLETE")
        self.log.info(f"  Charts  → {self.output_dir}/")
        self.log.info(f"  Findings→ {self.findings_dir}/")
        self.log.info("=" * 70)

        return {
            "panel":    panel_ry,
            "ranked":   ranked,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FITX Combined Roll Analysis")
    parser.add_argument("filepath", type=str, help="Path to FITX minute-bar CSV")
    parser.add_argument("--output-dir",   "-o", default="output_combined")
    args = parser.parse_args()

    pipeline = FITXRollPipeline(args.filepath, args.output_dir)
    results  = pipeline.run()
