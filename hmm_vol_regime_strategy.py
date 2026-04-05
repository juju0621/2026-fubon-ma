#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from hmmlearn import hmm as hmmlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING 
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"hmm_regime_{ts}.log"

    logger = logging.getLogger("HMMRegimeCTA")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    ))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log → %s", log_file)
    return logger


log = setup_logging()

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MULTIPLIER        = 200       # NT$ / 指數點
TICK_SIZE         = 1         # 最小升降單位

# 交易成本
TC_COMMISSION_RT  = 100       # NT$ RT / 口 (券商手續費 + 交易所費)
TC_TAX_RATE       = 0.00002   # 期交稅率 / 邊 (0.002%)
TC_SLIPPAGE_RT    = 200       # NT$ RT 滑點 (一般)
TC_SLIPPAGE_ROLL  = 100       # NT$ RT 滑點 (跨月價差單)

INITIAL_CAPITAL   = 5_000_000  # NT$

# 特徵工程
RESAMPLE_MIN      = 5
VOL_WIN_SHORT     = 6          # bars = 30 min
VOL_WIN_LONG      = 24         # bars = 2 h
MOMENTUM_WIN      = 12         # bars = 1 h
KF_OBS_NOISE      = 2.0
KF_PROC_NOISE     = 0.05

# HMM
N_STATES          = 4
HMM_COV           = "full"
HMM_ITER          = 300
HMM_SEED          = 42

# ── 進場門檻 ──────────────────────────────────────────────────────────────────
HMM_CONF_THRESHOLD   = 0.65   # HMM dominant state probability
REGIME_PERSIST_DAYS  = 2      # 機制持續天數 (含當日)
MOM_ZSCORE_THRESH    = 1.2    # 動量 z-score 閾值 (趨勢機制)
VWAP_MR_ZSCORE       = 1.6    # VWAP偏離 z-score 閾值 (CHAOS)
STORM_MAX_VWAP_DEV   = 0.004  # STORM 進場最大VWAP偏離 (0.4%)
STORM_MAX_VOL_EXP    = 2.0    # STORM 進場最大波動擴張
STORM_MIN_VOL_SURP   = -0.30  # STORM 進場最低成交量驚訝值

# ── 機制別倉位係數 ─────────────────────────────────────────────────────────────
SIZE_FACTOR: Dict[str, float] = {
    "BULL_QUIET": 1.00,
    "BEAR_QUIET": 1.00,
    "STORM"     : 0.60,
    "CHAOS"     : 0.40,
}
CARRY_BOOST       = 0.20      # carry對齊時倉位加成
CARRY_THRESHOLD   = 0.03      # |roll_yield| > 3% 才觸發carry調整

# ── 停損參數 (機制別) ──────────────────────────────────────────────────────────
ATR_STOP: Dict[str, float] = {
    "BULL_QUIET": 2.0,
    "BEAR_QUIET": 2.0,
    "STORM"     : 1.5,
    "CHAOS"     : 0.8,
}
CHAOS_PROFIT_TARGET_ATR = 1.5   # CHAOS 獲利目標 (正期望值)
TRAIL_MULT        = 1.5
ATR_PERIOD        = 14

DAILY_LOSS_LIM    = -0.015    # 日虧損限制 (NAV)
RISK_PER_TRADE    = 0.012
MAX_CONTRACTS     = 3
MAX_CHAOS_PER_DAY = 1         # 每日CHAOS交易上限

# ── 換倉 ──────────────────────────────────────────────────────────────────────
ROLL_DTE: Dict[str, Tuple[int, int]] = {
    "BULL_QUIET": (13, 17),
    "BEAR_QUIET": (0,  2),
    "STORM"     : (3,  8),
    "CHAOS"     : (3,  8),
}
ROLL_WINDOW = ("10:00", "11:00")

# ── Walk-forward ──────────────────────────────────────────────────────────────
WF_TRAIN_DAYS = 63
WF_TEST_DAYS  = 21
WF_STEP_DAYS  = 21

# ── EWM動量 ───────────────────────────────────────────────────────────────────
CTA_FAST_EWM  = 5
CTA_SLOW_EWM  = 20
MIN_HOLD_DAYS = 3

# ── 日內執行參數 ──────────────────────────────────────────────────────────────
EXEC_RESAMPLE_MIN = 10     # 執行頻率 (分鐘)：每 N 分鐘 K 棒才允許進場評估
MIN_HOLD_BARS     = 2      # 相同方向最短持倉執行棒數 (intraday anti-whipsaw)
SESSION_START     = "09:15"  # 最早進場時間
SESSION_END       = "13:15"  # 最晚進場時間


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class Regime(str, Enum):
    BULL_QUIET = "BULL_QUIET"
    BEAR_QUIET = "BEAR_QUIET"
    STORM      = "STORM"
    CHAOS      = "CHAOS"
    UNKNOWN    = "UNKNOWN"


class Signal(int, Enum):
    LONG  =  1
    FLAT  =  0
    SHORT = -1


@dataclass
class TCBreakdown:
    commission: float = 0.0
    tax       : float = 0.0
    slippage  : float = 0.0

    @property
    def total(self) -> float:
        return self.commission + self.tax + self.slippage

    def __add__(self, other: "TCBreakdown") -> "TCBreakdown":
        return TCBreakdown(
            commission = self.commission + other.commission,
            tax        = self.tax        + other.tax,
            slippage   = self.slippage   + other.slippage,
        )


@dataclass
class TradeRecord:
    entry_dt      : pd.Timestamp
    exit_dt       : Optional[pd.Timestamp]
    direction     : Signal
    entry_price   : float
    contracts     : int
    regime_entry  : Regime
    regime_exit   : Regime = Regime.UNKNOWN
    exit_price    : float  = 0.0
    gross_pnl     : float  = 0.0
    net_pnl       : float  = 0.0
    carry_pnl     : float  = 0.0
    tc_entry      : TCBreakdown = field(default_factory=TCBreakdown)
    tc_exit       : TCBreakdown = field(default_factory=TCBreakdown)
    exit_reason   : str    = ""
    alpha_engine  : str    = ""
    # 進場時的特徵快照 (for analysis)
    entry_conf    : float  = 0.0   # HMM dominant state probability
    entry_mom_z   : float  = 0.0   # momentum z-score at entry

    @property
    def tc(self) -> TCBreakdown:
        return self.tc_entry + self.tc_exit

    def close(self, exit_dt: pd.Timestamp, exit_price: float,
              reason: str, tc_exit: TCBreakdown, regime_exit: Regime):
        self.exit_dt     = exit_dt
        self.exit_price  = exit_price
        self.exit_reason = reason
        self.tc_exit     = tc_exit
        self.regime_exit = regime_exit
        self.gross_pnl   = (
            (exit_price - self.entry_price)
            * self.direction.value
            * MULTIPLIER * self.contracts
        )
        self.net_pnl = self.gross_pnl - self.tc.total

    def accumulate_carry(self, roll_yield: float,
                         bar_minutes: int = RESAMPLE_MIN):
        bar_frac        = bar_minutes / (252.0 * 390.0)
        self.carry_pnl += (roll_yield * bar_frac
                           * self.entry_price * MULTIPLIER
                           * self.contracts * self.direction.value)


@dataclass
class StrategyConfig:
    n_hmm_states      : int   = N_STATES
    resample_min      : int   = RESAMPLE_MIN
    vol_win_short     : int   = VOL_WIN_SHORT
    vol_win_long      : int   = VOL_WIN_LONG
    momentum_win      : int   = MOMENTUM_WIN
    atr_period        : int   = ATR_PERIOD
    trail_mult        : float = TRAIL_MULT
    daily_loss_limit  : float = DAILY_LOSS_LIM
    risk_per_trade    : float = RISK_PER_TRADE
    max_contracts     : int   = MAX_CONTRACTS
    use_pca           : bool  = True
    n_pca             : int   = 5
    # 日內執行參數
    exec_resample_min : int   = EXEC_RESAMPLE_MIN
    min_hold_bars     : int   = MIN_HOLD_BARS
    session_start     : str   = SESSION_START
    session_end       : str   = SESSION_END
    use_session_filter: bool  = True
    # 進場門檻
    hmm_conf_threshold  : float = HMM_CONF_THRESHOLD
    regime_persist_days : int   = REGIME_PERSIST_DAYS
    mom_zscore_thresh   : float = MOM_ZSCORE_THRESH
    vwap_mr_zscore      : float = VWAP_MR_ZSCORE
    storm_max_vwap_dev  : float = STORM_MAX_VWAP_DEV
    storm_max_vol_exp   : float = STORM_MAX_VOL_EXP
    storm_min_vol_surp  : float = STORM_MIN_VOL_SURP
    max_chaos_per_day   : int   = MAX_CHAOS_PER_DAY
    flat_before_close   : int   = 15   # 只用於 CHAOS (均值回歸需EOD平倉)


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTION COST MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionCostModel:
    @staticmethod
    def compute(price: float, contracts: int,
                is_roll: bool = False) -> TCBreakdown:
        commission = TC_COMMISSION_RT * contracts
        tax        = TC_TAX_RATE * 2 * price * MULTIPLIER * contracts
        slippage   = (TC_SLIPPAGE_ROLL if is_roll else TC_SLIPPAGE_RT) * contracts
        return TCBreakdown(commission=commission, tax=tax, slippage=slippage)


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRACT CALENDAR
# ═══════════════════════════════════════════════════════════════════════════════

class ContractCalendar:
    def __init__(self, start_year: int = 2023, end_year: int = 2027):
        self._exp: Dict[str, pd.Timestamp] = {}
        self._sorted: List[pd.Timestamp]   = []
        for yr in range(start_year, end_year + 1):
            for mo in range(1, 13):
                key = f"{yr}{mo:02d}"
                exp = self._third_wed(yr, mo)
                self._exp[key] = exp
                self._sorted.append(exp)
        self._sorted.sort()

    @staticmethod
    def _third_wed(year: int, month: int) -> pd.Timestamp:
        first  = pd.Timestamp(year, month, 1)
        to_wed = (2 - first.weekday()) % 7
        return first + pd.Timedelta(days=to_wed) + pd.Timedelta(weeks=2)

    def expiry(self, code: str) -> Optional[pd.Timestamp]:
        return self._exp.get(code)

    def near_far(self, date: pd.Timestamp) -> Tuple[str, str]:
        future = [e for e in self._sorted if e >= date]
        if len(future) < 2:
            raise ValueError(f"Not enough expiries after {date}")
        return future[0].strftime("%Y%m"), future[1].strftime("%Y%m")

    def dte(self, date: pd.Timestamp, code: str) -> int:
        exp = self._exp.get(code)
        return max(0, len(pd.bdate_range(date, exp)) - 1) if exp else 9999


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.outright : Optional[pd.DataFrame] = None
        self.spread   : Optional[pd.DataFrame] = None

    def load(self) -> "DataLoader":
        log.info("Loading %s …", self.csv_path)
        df = pd.read_csv(self.csv_path)
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
        df["datetime"] = pd.to_datetime(
            df["Txdate"].astype(str) + " " + df["BTime"].astype(str)
        )
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        for col in ["Px_O","Px_H","Px_L","Px_C"]:
            df[col] = df[col].replace(0.0, np.nan)
        is_spread     = df["Seccode"].str.contains("/")
        self.outright = df[~is_spread].copy()
        self.spread   = df[is_spread].copy()
        log.info("  outright=%d  spread=%d", len(self.outright), len(self.spread))
        return self

    def contract_bars(self, code: str) -> pd.DataFrame:
        sub = self.outright[self.outright["Seccode"] == f"FITX_{code}"].copy()
        sub.set_index("datetime", inplace=True)
        return sub

    def spread_bars(self, near: str, far: str) -> pd.DataFrame:
        key = f"FITX_{near}/{far}"
        sub = self.spread[(self.spread["Seccode"] == key)
                          & (self.spread["Trade_Vol"] > 0)].copy()
        if not sub.empty:
            sub.set_index("datetime", inplace=True)
        return sub


# ═══════════════════════════════════════════════════════════════════════════════
# KALMAN SMOOTHER
# ═══════════════════════════════════════════════════════════════════════════════

class KalmanSmoother:
    def __init__(self, obs=KF_OBS_NOISE, proc=KF_PROC_NOISE):
        self.R  = np.array([[obs**2]])
        q       = proc**2
        self.Q  = np.diag([q, q * 0.1])
        self.F  = np.array([[1., 1.], [0., 1.]])
        self.H  = np.array([[1., 0.]])
        self.P0 = np.eye(2) * 100.0

    def smooth(self, prices: np.ndarray) -> np.ndarray:
        out = np.empty(len(prices))
        x   = np.array([prices[0] if not np.isnan(prices[0]) else 0.0, 0.0])
        P   = self.P0.copy()
        for t, y in enumerate(prices):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            if not np.isnan(y):
                S = self.H @ P @ self.H.T + self.R
                K = P @ self.H.T @ np.linalg.inv(S)
                x = x + K.flatten() * (y - (self.H @ x)[0])
                P = (np.eye(2) - K @ self.H) @ P
            out[t] = x[0]
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# CONTINUOUS CONTRACT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousContractBuilder:
    def __init__(self, loader: DataLoader, cal: ContractCalendar):
        self.loader = loader
        self.cal    = cal

    def build(self) -> pd.DataFrame:
        log.info("Building continuous front-month series …")
        out = self.loader.outright.copy()
        out["month_code"] = out["Seccode"].str.extract(r"FITX_(\d{6})")
        pieces = []
        for d in sorted(out["Txdate"].unique()):
            dt     = pd.Timestamp(d)
            nc, fc = self.cal.near_far(dt)
            day    = out[(out["Txdate"] == d) & (out["month_code"] == nc)].copy()
            if day.empty:
                continue
            day = day.rename(columns={
                "Px_O":"open","Px_H":"high","Px_L":"low","Px_C":"close",
                "Px_Avg":"vwap","PX_Twap":"twap",
                "Trade_Vol":"volume","Trade_Cnt":"trade_cnt",
            })[["datetime","open","high","low","close","vwap","twap",
                "volume","trade_cnt"]].copy()
            day["near_code"] = nc
            day["far_code"]  = fc
            day["dte"]       = self.cal.dte(dt, nc)
            for c in ["open","high","low","close","vwap","twap"]:
                day[c] = day[c].ffill()
            pieces.append(day)

        df = pd.concat(pieces, ignore_index=True)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        log.info("  Continuous: %d bars  (%s → %s)",
                 len(df), df.index[0].date(), df.index[-1].date())
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEER  (14 features)
# ═══════════════════════════════════════════════════════════════════════════════

FEAT_COLS = [
    "log_ret_kf",    # F01  Kalman平滑5分鐘對數報酬
    "rv_short",      # F02  30分鐘實現波動率
    "vol_expansion", # F03  rv_short/rv_long  (機制轉換信號)
    "momentum",      # F04  1小時累積報酬
    "vwap_dev",      # F05  (close-VWAP)/close
    "range_norm",    # F06  (H-L)/close  (ATR代理)
    "vol_surprise",  # F07  log(成交量/EWM均量)
    "roll_yield",    # F08  年化 ln(近月/遠月)
    "liq_ratio",     # F09  遠月量/近月量 (換倉壓力)
    "time_sin",      # F10  日內時間循環編碼
    "time_cos",      # F11
    "spread_ar1",    # F12  roll yield AR(1)
    "mom_reversal",  # F13  短長期動量不一致 (潛在反轉)
    "garch_vol",     # F14  EWM-GARCH條件波動率
]


class FeatureEngineer:
    def __init__(self, loader: DataLoader, cal: ContractCalendar,
                 cfg: StrategyConfig, ks: KalmanSmoother):
        self.loader = loader
        self.cal    = cal
        self.cfg    = cfg
        self.ks     = ks

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        freq = f"{self.cfg.resample_min}min"
        agg  = {"open":"first","high":"max","low":"min","close":"last",
                "vwap":"mean","volume":"sum","trade_cnt":"sum",
                "near_code":"last","far_code":"last","dte":"last"}
        return df.resample(freq, closed="left", label="left").agg(agg
               ).dropna(subset=["close"])

    def _roll_yield(self, df: pd.DataFrame) -> pd.Series:
        result = pd.Series(np.nan, index=df.index, dtype=float)
        freq   = f"{self.cfg.resample_min}min"
        for nc in df["near_code"].unique():
            for fc in df.loc[df["near_code"]==nc,"far_code"].unique():
                mask = (df["near_code"]==nc) & (df["far_code"]==fc)
                if not mask.any(): continue
                ne, fe = self.cal.expiry(nc), self.cal.expiry(fc)
                if ne is None or fe is None: continue
                ann = 252.0 / max(1, (fe - ne).days)
                far = self.loader.contract_bars(fc)
                if far.empty: continue
                far5 = (far["Px_C"].replace(0,np.nan)
                        .resample(freq,closed="left",label="left").last().ffill())
                np_ = df.loc[mask,"close"]
                fp  = far5.reindex(np_.index).ffill()
                v   = (fp > 0) & (np_ > 0)
                if v.any():
                    result.loc[mask & v] = (np.log(np_[v]/fp[v]) * ann).values
        return result.ffill().fillna(0.0)

    def _liq_ratio(self, df: pd.DataFrame) -> pd.Series:
        result = pd.Series(np.nan, index=df.index, dtype=float)
        freq   = f"{self.cfg.resample_min}min"
        for nc in df["near_code"].unique():
            for fc in df.loc[df["near_code"]==nc,"far_code"].unique():
                mask = (df["near_code"]==nc) & (df["far_code"]==fc)
                if not mask.any(): continue
                far = self.loader.contract_bars(fc)
                if far.empty: continue
                fv  = far["Trade_Vol"].resample(freq,closed="left",label="left").sum()
                nv  = df.loc[mask,"volume"]
                aligned = fv.reindex(nv.index).fillna(0)
                result.loc[mask] = (aligned / nv.replace(0,np.nan)).values
        return result.ffill().fillna(0.0).clip(0, 10)

    def build(self, bar_df: pd.DataFrame) -> pd.DataFrame:
        log.info("Engineering features …")
        df = self._resample(bar_df)

        df["close_kf"]    = self.ks.smooth(df["close"].values.astype(float))
        df["log_ret_kf"]  = np.log(df["close_kf"] / df["close_kf"].shift(1))
        df["rv_short"]    = (df["log_ret_kf"]
                             .rolling(self.cfg.vol_win_short, min_periods=3).std())
        rv_long           = (df["log_ret_kf"]
                             .rolling(self.cfg.vol_win_long, min_periods=6).std())
        df["vol_expansion"] = (df["rv_short"]/rv_long.replace(0,np.nan)).clip(0,5)

        mom_short          = df["log_ret_kf"].rolling(3, min_periods=2).sum()
        df["momentum"]     = df["log_ret_kf"].rolling(self.cfg.momentum_win).sum()
        df["mom_reversal"] = (df["momentum"] * mom_short).apply(
            lambda x: -np.sign(x)*min(abs(x),0.02) if not np.isnan(x) else 0.0)

        df["vwap_dev"]     = ((df["close_kf"]-df["vwap"])
                              / df["close_kf"].replace(0,np.nan))
        df["range_norm"]   = ((df["high"]-df["low"])
                              / df["close_kf"].replace(0,np.nan))
        ewm_vol            = df["volume"].ewm(span=48, min_periods=6).mean()
        df["vol_surprise"] = np.log((df["volume"]+1) / (ewm_vol+1))

        # EWM-GARCH (λ=0.94)
        sq = df["log_ret_kf"]**2
        df["garch_vol"]    = sq.ewm(span=int(2/(1-0.94)-1), min_periods=5).mean().apply(np.sqrt)

        df["roll_yield"]   = self._roll_yield(df)
        df["liq_ratio"]    = self._liq_ratio(df)

        mins              = df.index.hour*60 + df.index.minute
        frac              = (mins - 8*60 - 45) / 300.0
        df["time_sin"]    = np.sin(2*np.pi*frac)
        df["time_cos"]    = np.cos(2*np.pi*frac)
        df["spread_ar1"]  = df["roll_yield"].shift(1).fillna(0.0)

        df.dropna(subset=FEAT_COLS, inplace=True)
        for c in FEAT_COLS:
            mu, sd = df[c].mean(), df[c].std()
            if sd > 0:
                df[c] = df[c].clip(mu - 3*sd, mu + 3*sd)

        keep = FEAT_COLS + ["close","close_kf","open","high","low",
                            "volume","dte","near_code","far_code"]
        log.info("  Feature matrix: %d rows × %d features", len(df), len(FEAT_COLS))
        return df[keep].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeClassifier:
    """
    4狀態 Gaussian HMM。

    標籤邏輯 (三軸):
      rv_short → 波動水平 (主要分割)
      momentum → 趨勢方向 (低波動狀態: 多頭/空頭)
      |momentum| → 方向性強弱 (高波動狀態: STORM vs CHAOS)

    低波動2態 → 動量正: BULL_QUIET, 動量負: BEAR_QUIET
    高波動2態 → |動量|較大: STORM, |動量|較小: CHAOS
    """

    def __init__(self, cfg: StrategyConfig):
        self.cfg             = cfg
        self.model           : Optional[hmmlib.GaussianHMM] = None
        self.scaler          = StandardScaler()
        self.pca             : Optional[PCA] = None
        self._state_regime   : Dict[int, Regime] = {}
        self._fitted         = False
        # 訓練期統計 (傳給Backtester作為進場門檻)
        self.train_mom_std   : float = 0.001
        self.train_vwap_std  : float = 0.001

    def _transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            Xs = self.scaler.fit_transform(X)
            if self.cfg.use_pca:
                n = min(self.cfg.n_pca, X.shape[1], X.shape[0]-1)
                self.pca = PCA(n_components=n, random_state=HMM_SEED)
                return self.pca.fit_transform(Xs)
            return Xs
        Xs = self.scaler.transform(X)
        return self.pca.transform(Xs) if self.pca else Xs

    @staticmethod
    def _day_lengths(idx: pd.DatetimeIndex) -> List[int]:
        days, lengths, count = None, [], 0
        for d in idx.normalize():
            if d == days:
                count += 1
            else:
                if count: lengths.append(count)
                count, days = 1, d
        if count: lengths.append(count)
        return lengths

    def _label_states(self, states: np.ndarray,
                      feat_df: pd.DataFrame) -> Dict[int, Regime]:
        rv_mean, mom_mean, abs_mom = {}, {}, {}
        for s in range(self.cfg.n_hmm_states):
            idx = np.where(states == s)[0]
            if len(idx) == 0:
                rv_mean[s] = np.inf; mom_mean[s] = 0.0; abs_mom[s] = 0.0
            else:
                rv_mean[s]  = float(feat_df["rv_short"].iloc[idx].mean())
                mom_mean[s] = float(feat_df["momentum"].iloc[idx].mean())
                abs_mom[s]  = float(feat_df["momentum"].iloc[idx].abs().mean())

        sorted_v = sorted(rv_mean, key=lambda k: rv_mean[k])
        n_low    = max(1, self.cfg.n_hmm_states // 2)
        low_s    = sorted_v[:n_low]
        high_s   = sorted_v[n_low:]

        mapping: Dict[int, Regime] = {}
        for s in low_s:
            mapping[s] = Regime.BULL_QUIET if mom_mean[s] >= 0 else Regime.BEAR_QUIET

        if len(high_s) == 1:
            mapping[high_s[0]] = Regime.STORM if abs_mom[high_s[0]] > 0 else Regime.CHAOS
        else:
            sorted_h = sorted(high_s, key=lambda k: abs_mom[k], reverse=True)
            mapping[sorted_h[0]] = Regime.STORM
            for s in sorted_h[1:]:
                mapping[s] = Regime.CHAOS

        log.debug("State→Regime: %s", {k: v.value for k, v in mapping.items()})
        log.debug("Vol/Mom per state: %s",
                  {k: f"rv={rv_mean[k]:.5f} mom={mom_mean[k]:.5f}"
                   for k in sorted(rv_mean)})
        return mapping

    def fit(self, feat_df: pd.DataFrame) -> "RegimeClassifier":
        X  = feat_df[FEAT_COLS].values.astype(float)
        Xt = self._transform(X, fit=True)
        L  = self._day_lengths(feat_df.index)

        # 記錄訓練期統計作為OOS門檻
        self.train_mom_std  = float(feat_df["momentum"].std()) or 0.001
        self.train_vwap_std = float(feat_df["vwap_dev"].std()) or 0.001

        best_score, best_model = -np.inf, None
        for trial in range(5):
            m = hmmlib.GaussianHMM(
                n_components=self.cfg.n_hmm_states,
                covariance_type=HMM_COV,
                n_iter=HMM_ITER,
                random_state=HMM_SEED + trial,
                verbose=False,
            )
            try:
                m.fit(Xt, lengths=L)
                sc = m.score(Xt, lengths=L)
                if sc > best_score:
                    best_score, best_model = sc, m
            except Exception:
                continue

        if best_model is None:
            raise RuntimeError("HMM failed to converge.")
        self.model         = best_model
        states             = self.model.predict(Xt, lengths=L)
        self._state_regime = self._label_states(states, feat_df)
        self._fitted       = True
        log.info("HMM fitted (LL=%.1f). Regime map: %s",
                 best_score, {k: v.value for k, v in self._state_regime.items()})
        return self

    def predict_sequence(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X  = feat_df[FEAT_COLS].values.astype(float)
        Xt = self._transform(X)
        L  = self._day_lengths(feat_df.index)
        st = self.model.predict(Xt, lengths=L)
        pr = self.model.predict_proba(Xt, lengths=L)

        out           = pd.DataFrame(index=feat_df.index)
        out["state"]  = st
        out["regime"] = [self._state_regime.get(s, Regime.UNKNOWN) for s in st]
        for i in range(self.cfg.n_hmm_states):
            out[f"prob_{i}"] = pr[:, i]
        # dominant state probability (信心度)
        out["confidence"] = pr.max(axis=1)
        # CHAOS state cumulative probability
        chaos_states      = [s for s,r in self._state_regime.items() if r==Regime.CHAOS]
        out["chaos_prob"] = (sum(pr[:,s] for s in chaos_states)
                             if chaos_states else np.zeros(len(pr)))
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# STOP LOSS MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class StopLossManager:
    def __init__(self, cfg: StrategyConfig):
        self.cfg              = cfg
        self.hard_stop        : Optional[float] = None
        self.trail_stop       : Optional[float] = None
        self.entry_px         : Optional[float] = None
        self.entry_atr        : Optional[float] = None
        self.direction        : Signal           = Signal.FLAT
        self._peak            : float            = 0.0
        self.stop_mult        : float            = 2.0
        self.profit_target_atr: Optional[float]  = None
        self.eod_flatten      : bool             = False   # 只有CHAOS開啟

    def arm(self, entry: float, atr: float, direction: Signal,
            stop_mult: float = 2.0,
            profit_target_atr: Optional[float] = None,
            eod_flatten: bool = False):
        self.entry_px          = entry
        self.entry_atr         = max(atr, float(TICK_SIZE))
        self.direction         = direction
        self._peak             = entry
        self.stop_mult         = stop_mult
        self.profit_target_atr = profit_target_atr
        self.eod_flatten       = eod_flatten
        offset = stop_mult * self.entry_atr
        if direction == Signal.LONG:
            self.hard_stop  = entry - offset
            self.trail_stop = None
        elif direction == Signal.SHORT:
            self.hard_stop  = entry + offset
            self.trail_stop = None

    def update_trail(self, price: float):
        if self.direction == Signal.FLAT or self.entry_atr is None:
            return
        trail  = self.cfg.trail_mult * self.entry_atr
        profit = (price - self.entry_px) * self.direction.value
        if self.direction == Signal.LONG:
            self._peak = max(self._peak, price)
            if profit >= self.entry_atr:
                nt = self._peak - trail
                self.trail_stop = max(self.trail_stop, nt) if self.trail_stop else nt
        else:
            self._peak = min(self._peak, price)
            if profit >= self.entry_atr:
                nt = self._peak + trail
                self.trail_stop = min(self.trail_stop, nt) if self.trail_stop else nt

    def check(self, o: float, h: float, l: float,
              dt: pd.Timestamp) -> Tuple[bool, float, str]:
        if self.direction == Signal.FLAT:
            return False, 0.0, ""

        # EOD強制平倉 (只有CHAOS才開啟)
        if self.eod_flatten:
            flat_dl = (pd.Timestamp(dt.date()).replace(hour=13, minute=45)
                       - pd.Timedelta(minutes=self.cfg.flat_before_close
                                      + self.cfg.resample_min))
            if dt >= flat_dl:
                return True, o, "eod_flatten"

        # 獲利目標 (CHAOS)
        if self.profit_target_atr and self.entry_atr:
            tgt = self.profit_target_atr * self.entry_atr
            if self.direction == Signal.LONG and h >= self.entry_px + tgt:
                return True, min(h, self.entry_px + tgt), "profit_target"
            if self.direction == Signal.SHORT and l <= self.entry_px - tgt:
                return True, max(l, self.entry_px - tgt), "profit_target"

        if self.direction == Signal.LONG:
            if l <= self.hard_stop:
                return True, min(o, self.hard_stop), "hard_stop"
            if self.trail_stop and l <= self.trail_stop:
                return True, min(o, self.trail_stop), "trail_stop"
        else:
            if h >= self.hard_stop:
                return True, max(o, self.hard_stop), "hard_stop"
            if self.trail_stop and h >= self.trail_stop:
                return True, max(o, self.trail_stop), "trail_stop"

        return False, 0.0, ""

    def disarm(self):
        self.hard_stop = self.trail_stop = None
        self.entry_px = self.entry_atr = None
        self.direction = Signal.FLAT
        self.profit_target_atr = None
        self.eod_flatten = False


# ═══════════════════════════════════════════════════════════════════════════════
# ROLLOVER MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class RolloverManager:
    def __init__(self, loader: DataLoader):
        self.loader  = loader
        self._rolled : set = set()

    def reset(self):
        self._rolled.clear()

    def should_roll(self, dt: pd.Timestamp, dte: int, near: str,
                    regime: Regime, spread_ar1: float) -> bool:
        if near in self._rolled: return False
        lo, hi = ROLL_DTE.get(regime.value, (3, 8))
        if not (lo <= dte <= hi): return False
        if regime == Regime.BULL_QUIET and spread_ar1 < 0: return False
        if regime == Regime.BEAR_QUIET and spread_ar1 > 0: return False
        return True

    @staticmethod
    def in_window(dt: pd.Timestamp) -> bool:
        t = dt.time()
        return (pd.Timestamp(ROLL_WINDOW[0]).time()
                <= t <=
                pd.Timestamp(ROLL_WINDOW[1]).time())

    def execute(self, dt: pd.Timestamp, near: str, far: str,
                contracts: int, price: float) -> TCBreakdown:
        spr = self.loader.spread_bars(near, far)
        use = (not spr.empty
               and np.abs((spr.index - dt).total_seconds()).min() <= 300)
        tc = TransactionCostModel.compute(price, contracts, is_roll=use)
        self._rolled.add(near)
        log.debug("Roll %s→%s | spread=%s | TC=%.0f", near, far, use, tc.total)
        return tc


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Bar-by-bar模擬，嚴格無未來資訊偷看。

    信號架構 (v2.2 日內化)
    ──────────────────────
    執行時機:   每 exec_resample_min 分鐘 K 棒（預設 30 分鐘）在交易時段內評估
                → 日內可觸發多次，上限由 min_hold_bars 反鞭刑控制
    方向決定:   每執行棒即時評估 5 分鐘 momentum z-score（無隔夜延遲）
    隔夜留倉:   BULL_QUIET / BEAR_QUIET / STORM 允許隔夜 (移除EOD強制平倉)
                CHAOS 仍保留EOD平倉 (均值回歸不適合留夜)

    進場門檻 (多重確認)
    ────────────────────
    1. HMM confidence > cfg.hmm_conf_threshold
    2. 機制持續性 ≥ cfg.regime_persist_days 天
    3. 動量z-score > cfg.mom_zscore_thresh  (趨勢機制)
       或 VWAP偏離z-score > cfg.vwap_mr_zscore  (CHAOS)
    4. STORM: |vwap_dev| < storm_max_vwap_dev
              vol_expansion < storm_max_vol_exp
              vol_surprise > storm_min_vol_surp

    出場邏輯
    ─────────
    1. ATR停損 (機制別)
    2. 追蹤停損 (獲利≥1ATR後啟動)
    3. 機制變換出場 → 當前機制 ≠ 進場機制
    4. 信號反轉 (趨勢方向改變)
    5. CHAOS: 獲利目標 1.5 ATR, EOD強制平倉
    6. 日虧損限制 (-1.5% NAV)
    """

    def __init__(self, feat_df: pd.DataFrame, clf: RegimeClassifier,
                 roll_mgr: RolloverManager, cfg: StrategyConfig,
                 capital: float = INITIAL_CAPITAL):
        self.feat_df     = feat_df.copy()
        self.clf         = clf
        self.roll        = roll_mgr
        self.cfg         = cfg
        self.tc_model    = TransactionCostModel()
        self.nav         = capital
        self._init_cap   = capital

        self.trades         : List[TradeRecord]   = []
        self._equity        : List[float]          = []
        self._eq_dt         : List[pd.Timestamp]   = []
        self._position      = Signal.FLAT
        self._contracts     = 0
        self._entry_regime  = Regime.UNKNOWN
        self._open_trade_obj: Optional[TradeRecord] = None
        self._sl            = StopLossManager(cfg)
        self._daily_nav     = capital
        self._daily_pnl     = 0.0
        self._last_date     : Optional[pd.Timestamp] = None
        self._pending_roll  = False
        self._chaos_today   = 0
        self._vwap_history  : List[float] = []

        # Intraday anti-whipsaw state
        self._last_exec_signal: Signal = Signal.FLAT  # last signal at exec bar
        self._last_exec_bar_i : int    = -9999         # bar index of last signal change

        # EWM動量信號 (從訓練期統計歸一化)
        self._mom_std   = clf.train_mom_std
        self._vwap_std  = clf.train_vwap_std

    # ─── ATR ──────────────────────────────────────────────────────────────────

    def _atr(self, idx: int) -> float:
        p   = self.cfg.atr_period
        sub = self.feat_df.iloc[max(0, idx-p):idx+1]
        if len(sub) < 2: return float(TICK_SIZE)
        hi, lo, cl = sub["high"].values, sub["low"].values, sub["close"].values
        tr = np.maximum(hi[1:]-lo[1:],
                        np.abs(hi[1:]-cl[:-1]),
                        np.abs(lo[1:]-cl[:-1]))
        return float(tr.mean()) if len(tr) else float(TICK_SIZE)

    def _size(self, regime: Regime, atr: float,
              roll_yield: float, direction: int) -> int:
        budget   = self.nav * self.cfg.risk_per_trade
        pts_risk = ATR_STOP.get(regime.value, 2.0) * max(atr, TICK_SIZE)
        base     = int(budget / (pts_risk * MULTIPLIER))
        factor   = SIZE_FACTOR.get(regime.value, 1.0)
        # Carry調整
        if abs(roll_yield) > CARRY_THRESHOLD:
            if (direction == 1 and roll_yield > 0) or (direction == -1 and roll_yield < 0):
                factor *= (1 + CARRY_BOOST)
            else:
                factor *= (1 - CARRY_BOOST)
        return max(1, min(int(base * factor), self.cfg.max_contracts))

    # ─── 進場過濾 ──────────────────────────────────────────────────────────────

    def _passes_entry_filter(
        self, regime: Regime, conf: float, chaos_prob: float,
        momentum: float, vwap_dev: float,
        vol_expansion: float, vol_surprise: float,
        signal: Signal
    ) -> Tuple[bool, str]:
        """
        Returns (passed, reject_reason).
        """
        # 1. HMM信心度
        if conf < self.cfg.hmm_conf_threshold:
            return False, f"conf={conf:.2f}<{self.cfg.hmm_conf_threshold}"

        # 2. 動量強度 (趨勢機制)
        if regime in (Regime.BULL_QUIET, Regime.BEAR_QUIET, Regime.STORM):
            mom_z = abs(momentum) / max(self._mom_std, 1e-8)
            if mom_z < self.cfg.mom_zscore_thresh:
                return False, f"mom_z={mom_z:.2f}<{self.cfg.mom_zscore_thresh}"

        # 3. STORM: 反追高殺低過濾
        if regime == Regime.STORM:
            if abs(vwap_dev) > self.cfg.storm_max_vwap_dev:
                return False, f"vwap_dev={vwap_dev:.4f}>threshold"
            if vol_expansion > self.cfg.storm_max_vol_exp:
                return False, f"vol_exp={vol_expansion:.2f}>max"
            if vol_surprise < self.cfg.storm_min_vol_surp:
                return False, f"vol_surp={vol_surprise:.2f}<min"
            # 方向需與momentum一致
            if signal == Signal.LONG  and momentum < 0: return False, "storm_dir_mismatch"
            if signal == Signal.SHORT and momentum > 0: return False, "storm_dir_mismatch"

        # 4. CHAOS: 只在波動收縮時進場 (fade the expansion)
        if regime == Regime.CHAOS:
            if vol_expansion > 1.2:
                return False, f"vol_exp={vol_expansion:.2f}>1.2"
            vwap_z = abs(vwap_dev) / max(self._vwap_std, 1e-8)
            if vwap_z < self.cfg.vwap_mr_zscore:
                return False, f"vwap_z={vwap_z:.2f}<{self.cfg.vwap_mr_zscore}"
            if self._chaos_today >= self.cfg.max_chaos_per_day:
                return False, f"chaos_limit={self._chaos_today}"

        return True, ""

    # ─── 機制持續性 ────────────────────────────────────────────────────────────

    def _is_regime_persistent(
        self, date: pd.Timestamp,
        daily_regimes: Dict[pd.Timestamp, Regime]
    ) -> bool:
        """連續 regime_persist_days 天同一機制才允許進場。"""
        dates  = sorted(daily_regimes.keys())
        target = daily_regimes.get(date)
        if target is None or target == Regime.UNKNOWN:
            return False
        idx = next((i for i, d in enumerate(dates) if d == date), None)
        if idx is None or idx < self.cfg.regime_persist_days - 1:
            return False
        for j in range(idx - self.cfg.regime_persist_days + 1, idx + 1):
            if daily_regimes[dates[j]] != target:
                return False
        return True

    # ─── 開倉 / 平倉 ──────────────────────────────────────────────────────────

    def _do_open(self, dt: pd.Timestamp, price: float, sig: Signal,
                 atr: float, regime: Regime, roll_yield: float,
                 conf: float, mom_z: float):
        n       = self._size(regime, atr, roll_yield, sig.value)
        tc_e    = self.tc_model.compute(price, n)
        self.nav -= tc_e.total
        self._position      = sig
        self._contracts     = n
        self._entry_regime  = regime

        stop_m  = ATR_STOP.get(regime.value, 2.0)
        pf_tgt  = CHAOS_PROFIT_TARGET_ATR if regime == Regime.CHAOS else None
        eod_flat= (regime == Regime.CHAOS)   # 只有CHAOS強制EOD平倉

        self._sl.arm(price, atr, sig,
                     stop_mult        = stop_m,
                     profit_target_atr= pf_tgt,
                     eod_flatten      = eod_flat)
        t = TradeRecord(
            entry_dt    = dt,
            exit_dt     = None,
            direction   = sig,
            entry_price = price,
            contracts   = n,
            regime_entry= regime,
            alpha_engine= regime.value,
            tc_entry    = tc_e,
            entry_conf  = conf,
            entry_mom_z = mom_z,
        )
        self._open_trade_obj = t
        self.trades.append(t)
        if regime == Regime.CHAOS: self._chaos_today += 1
        log.debug("OPEN %s ×%d @%.1f | regime=%s | conf=%.2f | mom_z=%.2f | TC=%.0f",
                  sig.name, n, price, regime.value, conf, mom_z, tc_e.total)

    def _do_close(self, dt: pd.Timestamp, price: float,
                  reason: str, regime: Regime):
        if self._open_trade_obj is None: return
        tc_x = self.tc_model.compute(price, self._contracts)
        self._open_trade_obj.close(dt, price, reason, tc_x, regime)
        self.nav         += self._open_trade_obj.gross_pnl
        self.nav         -= tc_x.total
        self._daily_pnl  += self._open_trade_obj.net_pnl
        self._position    = Signal.FLAT
        self._contracts   = 0
        self._entry_regime= Regime.UNKNOWN
        self._open_trade_obj = None
        self._sl.disarm()
        last = self.trades[-1]
        log.debug("CLOSE %s @%.1f | rsn=%s | gross=%.0f | net=%.0f",
                  reason, price, last.direction.name,
                  last.gross_pnl, last.net_pnl)

    def _mtm(self, price: float) -> float:
        if self._open_trade_obj is None: return 0.0
        return (
            (price - self._open_trade_obj.entry_price)
            * self._open_trade_obj.direction.value
            * MULTIPLIER * self._contracts
        )

    # ─── 每日主導機制 ─────────────────────────────────────────────────────────

    def _compute_daily_regime(
        self, sig_seq: pd.DataFrame
    ) -> Dict[pd.Timestamp, Regime]:
        """
        Returns {date → dominant_regime} from bar-level HMM predictions.
        Used only for the regime_persist check; no future data leakage.
        """
        daily_reg: Dict[pd.Timestamp, Regime] = {}
        for d in sig_seq.index.normalize().unique():
            day = sig_seq[sig_seq.index.normalize() == d]
            if day.empty:
                continue
            rc  = day["regime"].value_counts()
            daily_reg[pd.Timestamp(d.date())] = Regime(rc.index[0])
        return daily_reg

    # ─── 主循環 ────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        log.info("Backtesting %d bars …", len(self.feat_df))
        bars = self.feat_df
        self.roll.reset()
        self._open_trade_obj = None

        sig_seq   = self.clf.predict_sequence(bars)
        daily_reg = self._compute_daily_regime(sig_seq)

        # Pre-compute session time boundaries
        sess_start = pd.Timestamp(self.cfg.session_start).time()
        sess_end   = pd.Timestamp(self.cfg.session_end).time()
        exec_min   = self.cfg.exec_resample_min

        for i, (dt, row) in enumerate(bars.iterrows()):
            date = pd.Timestamp(dt.date())

            if date != self._last_date:
                self._daily_nav  = self.nav
                self._daily_pnl  = 0.0
                self._last_date  = date
                self._chaos_today = 0
                self._vwap_history.clear()

            o   = float(row["open"])
            h   = float(row["high"])
            l   = float(row["low"])
            c   = float(row["close"])
            dte = int(row["dte"])
            near= str(row["near_code"])
            far = str(row["far_code"])
            ry  = float(row["roll_yield"])
            vd  = float(row["vwap_dev"])
            mom = float(row["momentum"])
            ve  = float(row["vol_expansion"])
            vs  = float(row["vol_surprise"])
            atr = self._atr(i)

            self._vwap_history.append(vd)
            bar_regime = sig_seq["regime"].iloc[i]
            bar_conf   = float(sig_seq["confidence"].iloc[i])

            # 累計carry
            if self._open_trade_obj is not None:
                self._open_trade_obj.accumulate_carry(ry)

            # 追蹤停損更新
            if self._position != Signal.FLAT:
                self._sl.update_trail(c)

            # 停損/獲利目標 檢查
            stop_hit, stop_px, stop_rsn = self._sl.check(o, h, l, dt)
            if stop_hit and self._position != Signal.FLAT:
                self._do_close(dt, stop_px, stop_rsn, bar_regime)

            # ── 機制變換出場 ──────────────────────────────────────────────────
            if (self._position != Signal.FLAT
                    and self._entry_regime != Regime.UNKNOWN
                    and bar_regime != self._entry_regime
                    and bar_regime != Regime.UNKNOWN):
                self._do_close(dt, o, "regime_change", bar_regime)

            # ── 每日虧損限制 ──────────────────────────────────────────────────
            daily_pnl_pct = (self.nav - self._daily_nav) / max(self._daily_nav, 1)

            # ── 執行棒判斷：每 exec_resample_min 分鐘 + 交易時段過濾 ───────────
            is_exec_bar = (dt.minute % exec_min == 0)
            in_session  = (
                (dt.time() >= sess_start and dt.time() <= sess_end)
                if self.cfg.use_session_filter else True
            )

            if is_exec_bar and in_session and daily_pnl_pct > self.cfg.daily_loss_limit:
                mom_z = abs(mom) / max(self._mom_std, 1e-8)

                # ── 趨勢機制：BULL_QUIET / BEAR_QUIET / STORM ─────────────────
                if bar_regime in (Regime.BULL_QUIET, Regime.BEAR_QUIET, Regime.STORM):
                    # 日內方向由 5 分鐘 momentum 決定（無偷看）
                    if mom > self._mom_std * self.cfg.mom_zscore_thresh:
                        raw_sig: Signal = Signal.LONG
                    elif mom < -self._mom_std * self.cfg.mom_zscore_thresh:
                        raw_sig = Signal.SHORT
                    else:
                        raw_sig = Signal.FLAT

                    # 日內反鞭刑：min_hold_bars 執行棒內不允許反轉
                    if (raw_sig != Signal.FLAT
                            and raw_sig != self._last_exec_signal
                            and (i - self._last_exec_bar_i) < self.cfg.min_hold_bars):
                        raw_sig = self._last_exec_signal  # 維持前一方向

                    if raw_sig != Signal.FLAT:
                        if self._is_regime_persistent(date, daily_reg):
                            passed, reason = self._passes_entry_filter(
                                bar_regime, bar_conf, 0.0,
                                mom, vd, ve, vs, raw_sig
                            )
                            if passed:
                                if raw_sig != self._position:
                                    if self._position != Signal.FLAT:
                                        self._do_close(dt, o, "signal_reversal",
                                                       bar_regime)
                                    self._do_open(dt, o, raw_sig, atr, bar_regime,
                                                  ry, bar_conf, mom_z)
                                if raw_sig != self._last_exec_signal:
                                    self._last_exec_signal = raw_sig
                                    self._last_exec_bar_i  = i
                            else:
                                log.debug("Entry blocked: %s | %s",
                                          bar_regime.value, reason)

                # ── CHAOS 均值回歸 (日內執行棒觸發) ──────────────────────────
                elif bar_regime == Regime.CHAOS and self._position == Signal.FLAT:
                    vwap_z = abs(vd) / max(self._vwap_std, 1e-8)
                    mr_sig: Signal = (Signal.SHORT if vd > 0 else
                                      Signal.LONG  if vd < 0 else Signal.FLAT)
                    if mr_sig != Signal.FLAT:
                        passed, reason = self._passes_entry_filter(
                            Regime.CHAOS, bar_conf, 0.0,
                            mom, vd, ve, vs, mr_sig
                        )
                        if passed:
                            self._do_open(dt, o, mr_sig, atr, Regime.CHAOS,
                                          ry, bar_conf, vwap_z)

            # ── 換倉 ──────────────────────────────────────────────────────────
            if self._pending_roll and self.roll.in_window(dt):
                tc_r = self.roll.execute(dt, near, far, self._contracts, o)
                self.nav -= tc_r.total
                self._pending_roll = False

            if (not self._pending_roll
                    and self._position != Signal.FLAT
                    and self.roll.should_roll(dt, dte, near, bar_regime,
                                              float(row["spread_ar1"]))):
                self._pending_roll = True

            # Mark-to-market equity
            self._equity.append(self.nav + self._mtm(c))
            self._eq_dt.append(dt)

        # 期末平倉
        if self._open_trade_obj is not None:
            self._do_close(bars.index[-1], float(bars.iloc[-1]["close"]),
                           "end_of_period", Regime.UNKNOWN)

        eq_df = pd.DataFrame({"equity": self._equity}, index=self._eq_dt)
        log.info("  Trades: %d  |  Final NAV: %.0f TWD", len(self.trades), self.nav)
        return eq_df


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceAnalyzer:
    def __init__(self, eq_df: pd.DataFrame,
                 trades: List[TradeRecord],
                 init_cap: float = INITIAL_CAPITAL):
        self.eq       = eq_df["equity"].copy()
        self.trades   = trades
        self.init_cap = init_cap

    def total_return(self) -> float:
        return (self.eq.iloc[-1] - self.init_cap) / self.init_cap if not self.eq.empty else 0.0

    def daily_returns(self) -> pd.Series:
        return self.eq.resample("D").last().pct_change().dropna()

    def sharpe(self, rf=0.0) -> float:
        dr = self.daily_returns()
        if dr.empty or dr.std() == 0: return 0.0
        ex = dr - rf/252
        return float(ex.mean() / ex.std() * np.sqrt(252))

    def sortino(self, rf=0.0) -> float:
        dr = self.daily_returns(); ex = dr - rf/252
        ds = ex[ex < 0].std()
        return float(ex.mean()/ds*np.sqrt(252)) if ds > 0 else 0.0

    def max_drawdown(self) -> float:
        if self.eq.empty: return 0.0
        dd = (self.eq - self.eq.cummax()) / self.eq.cummax()
        return float(dd.min())

    def calmar(self) -> float:
        n   = max(len(self.daily_returns()), 1)
        ann = self.total_return() * 252 / n
        mdd = abs(self.max_drawdown())
        return ann / mdd if mdd > 0 else 0.0

    def win_rate(self) -> float:
        if not self.trades: return 0.0
        return sum(1 for t in self.trades if t.net_pnl > 0) / len(self.trades)

    def profit_factor(self) -> float:
        gw = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        gl = sum(abs(t.net_pnl) for t in self.trades if t.net_pnl < 0)
        return gw / gl if gl > 0 else float("inf")

    def avg_hold_days(self) -> float:
        durations = [(t.exit_dt - t.entry_dt).total_seconds()/86400
                     for t in self.trades if t.exit_dt]
        return float(np.mean(durations)) if durations else 0.0

    def attribution(self) -> Dict:
        if not self.trades: return {}
        long_t  = [t for t in self.trades if t.direction == Signal.LONG]
        short_t = [t for t in self.trades if t.direction == Signal.SHORT]

        def stats(tlist):
            if not tlist:
                return {"n":0,"gross":0.,"net":0.,"carry":0.,"tc":0.,
                        "win_rate":0.,"avg_gross":0.,"avg_hold":0.}
            return {
                "n"        : len(tlist),
                "gross"    : sum(t.gross_pnl for t in tlist),
                "net"      : sum(t.net_pnl   for t in tlist),
                "carry"    : sum(t.carry_pnl for t in tlist),
                "tc"       : sum(t.tc.total  for t in tlist),
                "win_rate" : sum(1 for t in tlist if t.net_pnl>0)/len(tlist),
                "avg_gross": sum(t.gross_pnl for t in tlist)/len(tlist),
                "avg_hold" : float(np.mean([(t.exit_dt-t.entry_dt).total_seconds()/3600
                                            for t in tlist if t.exit_dt]) or 0),
            }

        tg = sum(t.gross_pnl for t in self.trades)
        tc_breakdown = {
            "commission": sum(t.tc.commission for t in self.trades),
            "tax"       : sum(t.tc.tax        for t in self.trades),
            "slippage"  : sum(t.tc.slippage   for t in self.trades),
        }
        engine_pnl = {}
        for eng in set(t.alpha_engine for t in self.trades):
            et = [t for t in self.trades if t.alpha_engine == eng]
            engine_pnl[eng] = {
                "n"       : len(et),
                "gross"   : sum(t.gross_pnl for t in et),
                "net"     : sum(t.net_pnl   for t in et),
                "win_rate": sum(1 for t in et if t.net_pnl>0)/len(et),
            }
        return {
            "total_gross"    : tg,
            "total_carry"    : sum(t.carry_pnl for t in self.trades),
            "total_tc"       : sum(t.tc.total  for t in self.trades),
            "carry_pct_gross": sum(t.carry_pnl for t in self.trades)/tg if tg != 0 else 0.,
            "tc_pct_gross"   : sum(t.tc.total  for t in self.trades)/abs(tg) if tg != 0 else 0.,
            "long"           : stats(long_t),
            "short"          : stats(short_t),
            "by_engine"      : engine_pnl,
            "tc_breakdown"   : tc_breakdown,
        }

    def summary(self) -> Dict:
        return {
            "total_return"  : f"{self.total_return():.2%}",
            "sharpe"        : f"{self.sharpe():.3f}",
            "sortino"       : f"{self.sortino():.3f}",
            "calmar"        : f"{self.calmar():.3f}",
            "max_drawdown"  : f"{self.max_drawdown():.2%}",
            "n_trades"      : len(self.trades),
            "win_rate"      : f"{self.win_rate():.2%}",
            "profit_factor" : f"{self.profit_factor():.3f}",
            "avg_hold_hrs"  : f"{self.avg_hold_days()*24:.1f}h",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════

class WalkForwardAnalyzer:
    def __init__(self, feat_df: pd.DataFrame, loader: DataLoader,
                 cal: ContractCalendar, cfg: StrategyConfig,
                 train_days=WF_TRAIN_DAYS, test_days=WF_TEST_DAYS,
                 step=WF_STEP_DAYS):
        self.feat_df    = feat_df
        self.loader     = loader
        self.cal        = cal
        self.cfg        = cfg
        self.train_days = train_days
        self.test_days  = test_days
        self.step       = step
        self.fold_results : List[Dict]         = []
        self.all_equity   : List[pd.DataFrame] = []
        self.all_trades   : List[TradeRecord]  = []

    @staticmethod
    def _tdays(fd: pd.DataFrame) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(fd.index.normalize().unique()))

    @staticmethod
    def _slice(fd: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        m = (fd.index.normalize() >= s) & (fd.index.normalize() <= e)
        return fd[m]

    def run(self) -> pd.DataFrame:
        tdays  = self._tdays(self.feat_df)
        n      = len(tdays)
        cursor = 0; nav = INITIAL_CAPITAL; fold = 0

        log.info("Walk-forward: %d days | train=%d test=%d step=%d",
                 n, self.train_days, self.test_days, self.step)

        while cursor + self.train_days + self.test_days <= n:
            t0 = tdays[cursor]
            t1 = tdays[cursor + self.train_days - 1]
            t2 = tdays[cursor + self.train_days]
            t3 = tdays[min(cursor + self.train_days + self.test_days - 1, n - 1)]

            log.info("Fold %d | train %s→%s | test %s→%s",
                     fold+1, t0.date(), t1.date(), t2.date(), t3.date())

            train_df = self._slice(self.feat_df, t0, t1)
            test_df  = self._slice(self.feat_df, t2, t3)

            if len(train_df) < 200 or len(test_df) < 10:
                cursor += self.step; fold += 1; continue

            clf = RegimeClassifier(self.cfg)
            try:
                clf.fit(train_df)
            except Exception as exc:
                log.warning("Fold %d HMM failed: %s", fold+1, exc)
                cursor += self.step; fold += 1; continue

            bt = Backtester(
                feat_df = test_df,
                clf     = clf,
                roll_mgr= RolloverManager(self.loader),
                cfg     = self.cfg,
                capital = nav,
            )
            eq_df = bt.run()
            if eq_df.empty:
                cursor += self.step; fold += 1; continue

            perf = PerformanceAnalyzer(eq_df, bt.trades, nav)
            attr = perf.attribution()
            self.fold_results.append({
                "fold"       : fold+1,
                "train_start": str(t0.date()),
                "train_end"  : str(t1.date()),
                "test_start" : str(t2.date()),
                "test_end"   : str(t3.date()),
                **perf.summary(),
                "carry_pct"  : f"{attr.get('carry_pct_gross',0):.2%}",
                "tc_pct"     : f"{attr.get('tc_pct_gross',0):.2%}",
            })
            self.all_equity.append(eq_df)
            self.all_trades.extend(bt.trades)
            nav = bt.nav; cursor += self.step; fold += 1

        if not self.all_equity:
            log.warning("No folds completed.")
            return pd.DataFrame()

        combined = pd.concat(self.all_equity).sort_index()
        log.info("Walk-forward complete. Folds:%d  Trades:%d",
                 len(self.fold_results), len(self.all_trades))
        return combined


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISER
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {"BULL_QUIET":"#1976D2","BEAR_QUIET":"#EF5350",
          "STORM":"#FF9800","CHAOS":"#7B1FA2","UNKNOWN":"#9E9E9E"}


class StrategyVisualizer:
    def __init__(self, out_dir: str):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def plot_equity(self, eq: pd.DataFrame, folds: List[Dict]):
        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(3, 1, height_ratios=[3,1,1], hspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        e   = eq["equity"]
        dd  = (e - e.cummax()) / e.cummax()
        dr  = e.resample("D").last().pct_change().dropna()
        rs  = dr.rolling(63).apply(lambda x: x.mean()/x.std()*np.sqrt(252)
                                    if x.std() > 0 else 0)

        ax1.plot(e.index, e/1e6, "#1976D2", lw=1.2, label="NAV (M TWD)")
        ax1.fill_between(e.index, e/1e6, e.cummax()/1e6,
                         alpha=0.2, color="#EF5350", label="Drawdown")
        for r in folds:
            ax1.axvline(pd.Timestamp(r["test_start"]),
                        color="grey", lw=0.4, ls="--", alpha=0.4)
        ax1.set_ylabel("NAV (M TWD)")
        ax1.set_title("FITX VolRegime CTA v2.1 — Walk-Forward OOS Equity", fontsize=13)
        ax1.legend(fontsize=9); ax1.grid(alpha=0.2)
        ax2.fill_between(dd.index, dd*100, color="#EF5350", alpha=0.6)
        ax2.set_ylabel("DD %"); ax2.set_ylim(None, 0); ax2.grid(alpha=0.2)
        ax3.plot(rs.index, rs, "#43A047", lw=1)
        ax3.axhline(0, color="black", lw=0.5)
        ax3.axhline(1, color="#43A047", lw=0.5, ls="--", alpha=0.5)
        ax3.set_ylabel("Rolling Sharpe\n(63d)"); ax3.set_xlabel("Date")
        ax3.grid(alpha=0.2)
        plt.savefig(self.out/"01_equity_curve.png", dpi=150, bbox_inches="tight")
        plt.close(); log.info("Saved 01_equity_curve.png")

    def plot_regime_overlay(self, feat_df: pd.DataFrame, sig_df: pd.DataFrame):
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        daily_cl  = feat_df["close"].resample("D").last().dropna()
        axes[0].plot(daily_cl.index, daily_cl, "#212121", lw=0.8)
        axes[0].set_ylabel("FITX Close"); axes[0].grid(alpha=0.2)
        axes[0].set_title("FITX Price + HMM Regime Overlay")
        daily_reg = sig_df["regime"].resample("D").agg(
            lambda x: x.value_counts().index[0] if len(x) else "UNKNOWN")
        for dt, reg in daily_reg.items():
            axes[0].axvspan(dt, dt+pd.Timedelta(days=1),
                            alpha=0.12, color=COLORS.get(reg, "#9E9E9E"))
        daily_chaos = sig_df["chaos_prob"].resample("D").mean().dropna()
        axes[1].fill_between(daily_chaos.index, daily_chaos,
                             color="#7B1FA2", alpha=0.5, label="P(CHAOS)")
        axes[1].axhline(HMM_CONF_THRESHOLD, color="#FF9800", ls="--",
                        lw=1, label=f"Conf threshold ({HMM_CONF_THRESHOLD})")
        axes[1].set_ylabel("Chaos Probability"); axes[1].set_ylim(0, 1)
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.2)
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=c, label=r, alpha=0.6)
                   for r, c in COLORS.items()]
        axes[0].legend(handles=patches, fontsize=8, loc="upper left")
        plt.tight_layout()
        plt.savefig(self.out/"02_regime_overlay.png", dpi=150, bbox_inches="tight")
        plt.close(); log.info("Saved 02_regime_overlay.png")

    def plot_long_short(self, trades: List[TradeRecord]):
        if not trades: return
        long_t  = [t for t in trades if t.direction == Signal.LONG]
        short_t = [t for t in trades if t.direction == Signal.SHORT]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        def cumplot(tlist, label, ax, color):
            if not tlist: ax.axis("off"); return
            st = sorted(tlist, key=lambda x: x.entry_dt)
            cg = np.cumsum([t.gross_pnl for t in st])
            cn = np.cumsum([t.net_pnl   for t in st])
            cc = np.cumsum([t.carry_pnl for t in st])
            x  = range(len(cg))
            ax.plot(x, cg/1e3, color=color, lw=1.5, label="Gross P&L")
            ax.plot(x, cn/1e3, color=color, lw=1.0, ls="--", label="Net P&L")
            ax.fill_between(x, cc/1e3, alpha=0.3, color="#FF9800", label="Carry")
            ax.set_title(f"{label} Cumulative P&L (K NT$)")
            ax.set_xlabel("Trade #"); ax.legend(fontsize=8); ax.grid(alpha=0.2)

        cumplot(long_t,  "Long",  axes[0][0], "#1976D2")
        cumplot(short_t, "Short", axes[0][1], "#EF5350")

        # P&L distribution
        if long_t or short_t:
            axes[0][2].hist([t.net_pnl/1e3 for t in long_t],
                            bins=25, alpha=0.5, color="#1976D2", label="Long", density=True)
            axes[0][2].hist([t.net_pnl/1e3 for t in short_t],
                            bins=25, alpha=0.5, color="#EF5350", label="Short", density=True)
            axes[0][2].axvline(0, color="black", lw=1)
            axes[0][2].set_title("Net P&L Distribution (K NT$)")
            axes[0][2].legend(); axes[0][2].grid(alpha=0.2)

        # Attribution
        for idx, (tlist, label, color) in enumerate([
            (long_t, "Long",  "#1976D2"),
            (short_t, "Short", "#EF5350"),
            (trades, "Total",  "#43A047"),
        ]):
            ax = axes[1][idx]
            if not tlist: ax.axis("off"); continue
            grs = sum(t.gross_pnl for t in tlist)
            car = sum(t.carry_pnl for t in tlist)
            tcc = sum(t.tc.total  for t in tlist)
            vals = [(grs-car)/1e3, car/1e3, -tcc/1e3]
            lbis = ["Price Momentum", "Roll Carry", "Transaction Cost"]
            clrs = [color, "#FF9800", "#B71C1C"]
            bars_b = ax.bar(lbis, vals, color=clrs, alpha=0.8, edgecolor="white")
            ax.axhline(0, color="black", lw=0.5)
            ax.set_title(f"{label} P&L Attribution (K NT$)"); ax.grid(alpha=0.2, axis="y")
            for bar, v in zip(bars_b, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v,
                        f"{v:.0f}K", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8)

        plt.suptitle("Long/Short P&L Decomposition", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(self.out/"03_long_short_decomp.png", dpi=150, bbox_inches="tight")
        plt.close(); log.info("Saved 03_long_short_decomp.png")

    def plot_roll_yield(self, trades: List[TradeRecord], feat_df: pd.DataFrame):
        if not trades or feat_df.empty: return
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Carry vs Gross scatter
        cv = [t.carry_pnl/1e3 for t in trades if t.exit_dt]
        gv = [t.gross_pnl/1e3 for t in trades if t.exit_dt]
        if cv:
            axes[0].scatter(cv, gv, alpha=0.3, s=15,
                            c=["#1976D2" if t.direction==Signal.LONG else "#EF5350"
                               for t in trades if t.exit_dt])
            if len(cv) > 2:
                m, b = np.polyfit(cv, gv, 1)
                xs = np.linspace(min(cv), max(cv), 100)
                axes[0].plot(xs, m*xs+b, "k--", lw=1, label=f"β={m:.2f}")
            axes[0].axhline(0, color="k", lw=0.5); axes[0].axvline(0, color="k", lw=0.5)
            axes[0].set_xlabel("Carry P&L (K NT$)"); axes[0].set_ylabel("Gross P&L (K NT$)")
            axes[0].set_title("Carry vs Price Momentum (per trade)")
            axes[0].legend(); axes[0].grid(alpha=0.2)

        # Monthly heatmap
        if trades:
            tdf = pd.DataFrame([{"date": t.entry_dt, "net":t.net_pnl,"carry":t.carry_pnl}
                                 for t in trades if t.exit_dt])
            if not tdf.empty:
                tdf.set_index("date", inplace=True)
                mp = tdf["net"].resample("ME").sum()
                mc = tdf["carry"].resample("ME").sum()
                x  = range(len(mp))
                axes[1].bar(x, mp.values/1e3,
                            color=["#43A047" if v>0 else "#EF5350" for v in mp.values],
                            alpha=0.7, label="Net P&L")
                axes[1].plot(x, mc.values/1e3, "o--", color="#FF9800",
                             markersize=4, label="Carry")
                axes[1].set_xticks(x)
                axes[1].set_xticklabels([d.strftime("%y-%m") for d in mp.index],
                                        rotation=45, ha="right", fontsize=7)
                axes[1].set_title("Monthly P&L vs Carry")
                axes[1].legend(fontsize=8); axes[1].grid(alpha=0.2)

        # Roll yield time series
        ry = feat_df["roll_yield"].resample("D").mean().dropna()
        axes[2].fill_between(ry.index, ry*100, where=ry>0,
                             alpha=0.6, color="#43A047", label="Backwardation (short favorable)")
        axes[2].fill_between(ry.index, ry*100, where=ry<0,
                             alpha=0.6, color="#EF5350", label="Contango (long favorable?)")
        axes[2].axhline(0, color="black", lw=0.5)
        axes[2].set_ylabel("Annualized Roll Yield %")
        axes[2].set_title("FITX Roll Yield (Basis Structure)")
        axes[2].legend(fontsize=8); axes[2].grid(alpha=0.2)

        plt.tight_layout()
        plt.savefig(self.out/"04_roll_yield.png", dpi=150, bbox_inches="tight")
        plt.close(); log.info("Saved 04_roll_yield.png")

    def plot_tc_waterfall(self, trades: List[TradeRecord]):
        if not trades: return
        grs  = sum(t.gross_pnl for t in trades)
        comm = sum(t.tc.commission for t in trades)
        tax  = sum(t.tc.tax       for t in trades)
        slip = sum(t.tc.slippage  for t in trades)
        net  = sum(t.net_pnl      for t in trades)

        cats = ["Gross P&L", "Commission", "Futures Tax", "Slippage", "Net P&L"]
        vals = [grs, -comm, -tax, -slip, net]
        cols = ["#1976D2","#EF5350","#EF5350","#EF5350","#43A047" if net>0 else "#EF5350"]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(cats, [v/1e3 for v in vals], color=cols, alpha=0.85, edgecolor="white")
        ax.axhline(0, color="black", lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v/1e3,
                    f"{v/1e3:.1f}K", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
        pct = (comm+tax+slip)/abs(grs)*100 if grs != 0 else 0
        ax.annotate(f"TC Drag: {pct:.1f}% of |Gross P&L|",
                    xy=(0.98, 0.02), xycoords="axes fraction",
                    ha="right", fontsize=10, color="darkred")
        ax.set_ylabel("K NT$"); ax.set_title("P&L Waterfall: Gross → Net")
        ax.grid(alpha=0.2, axis="y")
        plt.tight_layout()
        plt.savefig(self.out/"05_tc_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close(); log.info("Saved 05_tc_waterfall.png")

    def plot_regime_perf(self, trades: List[TradeRecord]):
        if not trades: return
        regimes = [r.value for r in Regime if r != Regime.UNKNOWN]
        data = {}
        for reg in regimes:
            rt = [t for t in trades if t.regime_entry.value == reg]
            if rt:
                hold_hrs = [(t.exit_dt - t.entry_dt).total_seconds() / 3600
                            for t in rt if t.exit_dt]
                data[reg] = {
                    "n"        : len(rt),
                    "net"      : sum(t.net_pnl for t in rt) / 1e3,
                    "win_rate" : sum(1 for t in rt if t.net_pnl > 0) / len(rt),
                    "avg_net"  : sum(t.net_pnl for t in rt) / len(rt) / 1e3,
                    "hold_dist": hold_hrs,
                }
        if not data: return
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        regs = list(data.keys())
        clrs = [COLORS.get(r, "#9E9E9E") for r in regs]

        # First three sub-plots: bar charts
        bar_metrics = [("net", "Net P&L (K NT$)"),
                       ("win_rate", "Win Rate (%)"),
                       ("avg_net", "Avg Net/Trade (K NT$)")]
        for ax, (key, title) in zip(axes[:3], bar_metrics):
            vals = [data[r].get(key, 0) for r in regs]
            if key == "win_rate":
                vals = [v * 100 for v in vals]
            ax.bar(regs, vals, color=clrs, alpha=0.8, edgecolor="white")
            ax.axhline(0 if key != "win_rate" else 50, color="black", lw=0.5, ls="--")
            ax.set_title(title); ax.grid(alpha=0.2, axis="y")
            for j, (r, v) in enumerate(zip(regs, vals)):
                ax.text(j, v, f"n={data[r]['n']}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7)

        # Fourth sub-plot: hold-time distribution (box plot)
        ax4 = axes[3]
        box_data = [data[r]["hold_dist"] for r in regs]
        bp = ax4.boxplot(box_data, patch_artist=True, widths=0.5,
                         medianprops=dict(color="white", lw=2),
                         whiskerprops=dict(lw=1.2),
                         capprops=dict(lw=1.2),
                         flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch, clr in zip(bp["boxes"], clrs):
            patch.set_facecolor(clr); patch.set_alpha(0.8)
        for j, (r, d) in enumerate(zip(regs, box_data)):
            if d:
                med = float(np.median(d))
                ax4.text(j + 1, med, f"{med:.1f}h", ha="center",
                         va="bottom", fontsize=7, color="white", fontweight="bold")
        ax4.set_xticks(range(1, len(regs) + 1))
        ax4.set_xticklabels(regs)
        ax4.set_title("Hold Time Distribution (hrs)")
        ax4.set_ylabel("hrs")
        ax4.grid(alpha=0.2, axis="y")

        plt.suptitle("Performance by Regime", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.out/"06_regime_perf.png", dpi=150, bbox_inches="tight")
        plt.close(); log.info("Saved 06_regime_perf.png")

    def plot_monthly_heatmap(self, eq: pd.DataFrame):
        monthly = eq["equity"].resample("ME").last().pct_change().dropna()*100
        if len(monthly) < 2: return
        df_m = monthly.to_frame("ret")
        df_m["year"] = df_m.index.year; df_m["month"] = df_m.index.month
        pivot = df_m.pivot(index="year", columns="month", values="ret")
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
        fig, ax = plt.subplots(figsize=(14, max(3, len(pivot))))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=5)
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i,j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="Monthly Return %"); ax.set_title("Monthly Return Heatmap")
        plt.tight_layout()
        plt.savefig(self.out/"07_monthly_heatmap.png", dpi=120, bbox_inches="tight")
        plt.close(); log.info("Saved 07_monthly_heatmap.png")

    def plot_pca_analysis(self, clf: "RegimeClassifier", _feat_df: pd.DataFrame = None):
        """
        08_pca_analysis.png — three panels:
          (A) Scree plot: explained variance ratio per component + cumulative
          (B) Feature loadings heatmap (diverging): which original features
              each PC captures (rows=PCs, cols=features)
          (C) Weighted feature importance bar chart: each feature's
              contribution = Σ_k (explained_var_k × loading_k²), normalised
        """
        if clf.pca is None or clf.scaler is None:
            log.warning("PCA not available — skipping 08_pca_analysis.png")
            return

        pca  = clf.pca
        n_pc = pca.n_components_
        evr  = pca.explained_variance_ratio_          # shape (n_pc,)
        comp = pca.components_                         # shape (n_pc, n_feat)

        # Weighted feature importance: Σ_k evr_k × loading²_k  (normalised)
        importance = (evr[:, np.newaxis] * comp ** 2).sum(axis=0)
        importance /= importance.sum()
        feat_labels = FEAT_COLS

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle("PCA Dimensionality Reduction — Feature Importance & Component Structure", fontsize=13, y=1.02)

        # ── Panel A: Scree plot ────────────────────────────────────────────
        ax = axes[0]
        cumvar = np.cumsum(evr) * 100
        bars_a = ax.bar(range(1, n_pc + 1), evr * 100,
                        color="#1976D2", alpha=0.8, edgecolor="white",
                        label="Individual")
        ax.plot(range(1, n_pc + 1), cumvar, "o--",
                color="#EF5350", lw=1.5, ms=6, label="Cumulative")
        for bar, v in zip(bars_a, evr * 100):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.axhline(80, color="#FF9800", lw=0.8, ls="--", alpha=0.7,
                   label="80% threshold")
        ax.set_xlabel("Principal Component"); ax.set_ylabel("Explained Variance (%)")
        ax.set_title("(A) Scree Plot")
        ax.set_xticks(range(1, n_pc + 1))
        ax.set_xticklabels([f"PC{k}" for k in range(1, n_pc + 1)])
        ax.set_ylim(0, max(cumvar) * 1.12)
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

        # ── Panel B: Feature loadings heatmap ─────────────────────────────
        ax = axes[1]
        vmax = np.abs(comp).max()
        im = ax.imshow(comp, cmap="RdBu_r", aspect="auto",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(feat_labels)))
        ax.set_xticklabels(feat_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n_pc))
        ax.set_yticklabels([f"PC{k + 1}\n({evr[k]*100:.1f}%)"
                            for k in range(n_pc)], fontsize=8)
        for r in range(n_pc):
            for c_idx in range(len(feat_labels)):
                val = comp[r, c_idx]
                ax.text(c_idx, r, f"{val:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if abs(val) > vmax * 0.6 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Loading")
        ax.set_title("(B) Feature Loadings (Red=Positive, Blue=Negative)")

        # ── Panel C: Weighted feature importance ──────────────────────────
        ax = axes[2]
        order = np.argsort(importance)[::-1]
        clrs  = ["#1976D2" if importance[j] >= np.median(importance)
                 else "#90CAF9" for j in order]
        bars_c = ax.barh([feat_labels[j] for j in order],
                         importance[order] * 100,
                         color=clrs, edgecolor="white", alpha=0.9)
        for bar, v in zip(bars_c, importance[order] * 100):
            ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=7)
        ax.set_xlabel("Weighted Importance (%)\n(Σ evr_k × loading²_k)")
        ax.set_title("(C) Feature Importance Ranking")
        ax.set_xlim(0, importance.max() * 100 * 1.25)
        ax.grid(alpha=0.2, axis="x")

        plt.tight_layout()
        plt.savefig(self.out / "08_pca_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved 08_pca_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class StrategyRunner:
    def __init__(self, csv_path: str,
                 cfg: Optional[StrategyConfig] = None,
                 output_dir: str = "output/hmm_regime"):
        self.csv_path   = csv_path
        self.cfg        = cfg or StrategyConfig()
        self.output_dir = output_dir

    def run(self):
        log.info("═" * 70)
        log.info("FITX VolRegime CTA v2.2  |  Multi-Regime HMM Strategy (Intraday)")
        log.info("═" * 70)

        loader  = DataLoader(self.csv_path).load()
        cal     = ContractCalendar()
        bar_df  = ContinuousContractBuilder(loader, cal).build()
        feat_df = FeatureEngineer(loader, cal, self.cfg, KalmanSmoother()).build(bar_df)

        wfa    = WalkForwardAnalyzer(feat_df, loader, cal, self.cfg)
        oos_eq = wfa.run()

        if oos_eq.empty:
            log.error("No OOS equity data. Exiting.")
            return

        agg  = PerformanceAnalyzer(oos_eq, wfa.all_trades, INITIAL_CAPITAL)
        attr = agg.attribution()

        log.info("=== AGGREGATE OOS PERFORMANCE ===")
        for k, v in agg.summary().items():
            log.info("  %-22s : %s", k, v)
        log.info("=== P&L ATTRIBUTION ===")
        log.info("  Total gross:  NT$%.0f", attr.get("total_gross",0))
        log.info("  Carry P&L:    NT$%.0f  (%.2f%%)",
                 attr.get("total_carry",0), attr.get("carry_pct_gross",0)*100)
        log.info("  TC drag:      NT$%.0f  (%.2f%% of |gross|)",
                 attr.get("total_tc",0), attr.get("tc_pct_gross",0)*100)
        log.info("  Long net:     NT$%.0f  n=%d",
                 attr.get("long",{}).get("net",0), attr.get("long",{}).get("n",0))
        log.info("  Short net:    NT$%.0f  n=%d",
                 attr.get("short",{}).get("net",0), attr.get("short",{}).get("n",0))

        # Full-sample HMM for visualisation
        log.info("Full-sample HMM for visualisation …")
        full_clf = RegimeClassifier(self.cfg)
        try:
            full_clf.fit(feat_df)
            sig_df = full_clf.predict_sequence(feat_df)
        except Exception as exc:
            log.warning("Full-sample HMM: %s", exc)
            sig_df = pd.DataFrame(
                {"regime": Regime.UNKNOWN, "confidence": 0.0, "chaos_prob": 0.0},
                index=feat_df.index)

        vis = StrategyVisualizer(self.output_dir)
        vis.plot_equity(oos_eq, wfa.fold_results)
        vis.plot_regime_overlay(feat_df, sig_df)
        vis.plot_long_short(wfa.all_trades)
        vis.plot_roll_yield(wfa.all_trades, feat_df)
        vis.plot_tc_waterfall(wfa.all_trades)
        vis.plot_regime_perf(wfa.all_trades)
        vis.plot_monthly_heatmap(oos_eq)
        vis.plot_pca_analysis(full_clf, feat_df)

        out = Path(self.output_dir)
        oos_eq.to_csv(out/"equity_curve.csv")
        pd.DataFrame(wfa.fold_results).to_csv(out/"fold_results.csv", index=False)

        rows = []
        for t in wfa.all_trades:
            rows.append({
                "entry_dt"    : t.entry_dt,
                "exit_dt"     : t.exit_dt,
                "direction"   : t.direction.name,
                "entry_price" : t.entry_price,
                "exit_price"  : t.exit_price,
                "contracts"   : t.contracts,
                "regime_entry": t.regime_entry.value,
                "regime_exit" : t.regime_exit.value,
                "alpha_engine": t.alpha_engine,
                "gross_pnl"   : round(t.gross_pnl,2),
                "net_pnl"     : round(t.net_pnl,2),
                "carry_pnl"   : round(t.carry_pnl,2),
                "tc_commission": round(t.tc.commission,2),
                "tc_tax"      : round(t.tc.tax,2),
                "tc_slippage" : round(t.tc.slippage,2),
                "tc_total"    : round(t.tc.total,2),
                "exit_reason" : t.exit_reason,
                "entry_conf"  : round(t.entry_conf,3),
                "entry_mom_z" : round(t.entry_mom_z,3),
            })
        pd.DataFrame(rows).to_csv(out/"trade_log.csv", index=False)
        log.info("CSVs saved → %s", out)


        log.info("═" * 70)
        log.info("Pipeline complete. Output → %s", self.output_dir)
        log.info("═" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FITX VolRegime CTA v2.2 (Intraday)")
    parser.add_argument("--csv",      default="data.csv")
    parser.add_argument("--outdir",   default="output/hmm_regime")
    parser.add_argument("--states",   type=int,   default=N_STATES)
    parser.add_argument("--pca",      type=int,   default=5)
    parser.add_argument("--conf",     type=float, default=HMM_CONF_THRESHOLD)
    parser.add_argument("--mom-z",    type=float, default=MOM_ZSCORE_THRESH)
    parser.add_argument("--exec-min", type=int,   default=EXEC_RESAMPLE_MIN,
                        help="Execution bar frequency in minutes (default: 30)")
    parser.add_argument("--sess-start", default=SESSION_START,
                        help="Session start time HH:MM (default: 09:15)")
    parser.add_argument("--sess-end",   default=SESSION_END,
                        help="Session end time HH:MM (default: 13:15)")
    args = parser.parse_args()

    cfg = StrategyConfig(
        n_hmm_states       = args.states,
        n_pca              = args.pca,
        hmm_conf_threshold = args.conf,
        mom_zscore_thresh  = args.mom_z,
        exec_resample_min  = args.exec_min,
        session_start      = args.sess_start,
        session_end        = args.sess_end,
    )
    StrategyRunner(args.csv, cfg, args.outdir).run()
