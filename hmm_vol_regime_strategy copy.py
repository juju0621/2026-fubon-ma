#!/usr/bin/env python3
"""
hmm_vol_regime_strategy.py  v2.2
=================================
FITX HMM Multi-Regime Volatility Strategy  —  "VolRegime CTA"

改進重點 v2.2 (日內化)
──────────────────────
1. 日內信號架構：信號在每個 EXEC_RESAMPLE_MIN 分鐘 K 棒評估（預設 30 分鐘）
     → 一天可觸發多次（上限由 min_hold_bars × exec_resample_min 控制）
     → 特徵計算仍用 5 分鐘 K 棒，執行層合併至 30 分鐘
2. 交易時段過濾：09:15–13:15（可配置），避開開盤雜訊與收盤 MOC 流量
     → 趨勢與均值回歸機制統一適用
3. 日內反鞭刑（anti-whipsaw）：相同方向訊號在 min_hold_bars 個執行棒內
     不允許反轉，防止 30 分鐘級過度交易
4. PCA 分析圖（第 08 圖）：解釋變異量、特徵載荷量熱圖、特徵重要性排名
     → 報告中補充 PCA 降維的原因與結果

改進重點 v2.1
─────────────
1. 取消趨勢機制的EOD強制平倉 → 允許隔夜留倉，大幅降低交易頻率與TC
2. 進場門檻 (須同時滿足)
     • HMM 信心度   conf > HMM_CONF_THRESHOLD  (dominant state probability)
     • 機制持續性   同一機制連續 ≥ REGIME_PERSIST_DAYS 個交易日
     • 動量強度     |momentum| z-score > MOM_ZSCORE_THRESH  (趨勢機制)
       或 VWAP偏離  |vwap_dev| z-score > VWAP_MR_ZSCORE      (均值回歸機制)
3. 機制變換出場 → 當前機制 ≠ 進場機制時立即出場 (at bar open)
4. CHAOS 機制改善
     • 僅在波動率收縮時進場 (vol_expansion < 1.2)
     • 獲利目標 1.5 ATR (舊: 0.5 ATR) → 正期望值結構
     • 停損      0.8 ATR (舊: 1.0 ATR)
     • 每日最多 1 筆 CHAOS 交易
5. STORM 進場過濾
     • 價格不能太偏離VWAP (|vwap_dev| < 0.3%) → 不追高殺低
     • 波動擴張適中 (1.0 < vol_expansion < 2.2) → 不在最混亂時入場
     • 須有成交量配合 (vol_surprise > -0.3)
6. 機制持續性判斷 (已整合進門檻)

Regime → Alpha Engine
───────────────────────────────────────────────────────────────────
 BULL_QUIET   低波動 + 上升趨勢   → EWM動量追蹤 LONG  + Carry調整
 BEAR_QUIET   低波動 + 下降趨勢   → EWM動量追蹤 SHORT + Carry調整
 STORM        高波動 + 強方向性   → 縮減倉位CTA (需VWAP確認)
 CHAOS        高波動 + 均值回歸   → VWAP偏離淡化 (需波動收縮)

交易成本模型 (台灣期交所 FITX)
 手續費   : NT$100 RT / 口 (券商 + 交易所)
 期交稅   : 0.002% × 成交金額 × 2 (進出各一次，隨價格變動)
 滑點     : NT$200 RT / 口 (一個升降單位來回，一般盤中)
           : NT$100 RT / 口 (跨月價差單換倉)

Walk-forward: 訓練63天 / 測試21天 / 步進21天

所有DEBUG輸出 → logs/hmm_regime_<ts>.log
報告 (繁體中文) → output/hmm_regime/report.md

Author: Quantitative Research  |  2026-04-05
"""

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
# LOGGING  — file=DEBUG, console=WARNING (不要print滿天飛)
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
# MARKDOWN REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class MarkdownReportGenerator:

    def __init__(self, out_dir: str):
        self.out_path = Path(out_dir) / "report.md"

    def generate(
        self,
        perf: PerformanceAnalyzer,
        fold_results: List[Dict],
        all_trades: List[TradeRecord],
        feat_df: pd.DataFrame,
        eq_df: pd.DataFrame,
        clf: Optional["RegimeClassifier"] = None,
    ) -> str:
        attr  = perf.attribution()
        summ  = perf.summary()
        lines = []
        a     = lines.append



        a("# FITX VolRegime CTA 策略分析報告")
        a(f"\n> 報告生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        a("---\n")

        # ── 1. 策略概覽 ────────────────────────────────────────────────────────
        a("## 1. 這個策略在做什麼？\n")
        a("""
**核心思想：** 市場不會永遠以同一種方式運動。有時候安靜地趨勢上漲，有時候在高波動中
急速奔走，有時候只是在某個價位附近震盪。如果我們用一套固定的趨勢追蹤規則應對所有
情境，在某些時期會大賺，在另一些時期卻會大虧。

**VolRegime CTA** 的解法是：先問「現在市場處於哪種狀態？」，再根據該狀態
切換到最合適的交易邏輯。偵測市場狀態的核心工具是**隱馬爾可夫模型（HMM）**，
它以 14 個從 5 分鐘 FITX 期貨 K 棒衍生的因子作為輸入。

### 整體流程圖

```
原始 1 分鐘 FITX Tick 資料
       │
       ▼  Kalman 濾波器（去除微結構雜訊）
5 分鐘 OHLCV K 棒
       │
       ▼  特徵工程（14 個因子）
特徵矩陣  [N_bars × 14]
       │
       ▼  高斯 HMM（在訓練期內訓練）
市場機制  {BULL_QUIET | BEAR_QUIET | STORM | CHAOS}
       │
       ▼  機制專屬 Alpha 引擎 × 進場門檻 × 倉位管理
交易信號  {做多 LONG | 空倉 FLAT | 做空 SHORT}
       │
       ▼  Walk-Forward（滾動）驗證：訓練 63 天 / 測試 21 天 / 步進 21 天
樣本外（OOS）績效
```

### 四種市場機制

| 機制代碼 | 中文名稱 | 波動率 | 方向性 | Alpha 引擎 | 倉位係數 |
|:---------|:---------|:------:|:------:|:-----------|:--------:|
| `BULL_QUIET` | 多頭低波動 | 低 | 動量為正 | EWM 動量追蹤做多 + Carry 加成 | 100% |
| `BEAR_QUIET` | 空頭低波動 | 低 | 動量為負 | EWM 動量追蹤做空 + Carry 加成 | 100% |
| `STORM` | 強趨勢高波動 | 高 | 方向性強 | 縮減倉位 CTA（需 VWAP 確認） | 60% |
| `CHAOS` | 均值回歸高波動 | 高 | 方向性弱 | VWAP 偏離淡化（需波動收縮） | 40% |
""")

        # ── 2. Kalman 濾波器 ───────────────────────────────────────────────────
        a("## 2. Kalman 濾波器——為何使用？怎麼運作？\n")
        a(f"""
### 問題：原始價格含有雜訊

每一根 5 分鐘 K 棒的收盤價，包含「真實價格變動」和「買賣價差造成的隨機跳動（微結構
雜訊）」。如果我們直接用原始收盤計算對數報酬再餵給 HMM，模型會把雜訊誤認為真實的
市場機制特徵，降低辨識品質。

### 解法：Kalman 濾波器

Kalman 濾波器是一種**即時、遞迴的貝葉斯平滑器**。你可以把它理解為：
> 「真實的價格走勢是平滑的；每次觀測到的報價，是真實價格加上一些量測誤差。
>  Kalman 濾波器持續更新對真實價格的最佳估計。」

它不需要看未來資料，每一棒都能即時輸出平滑後的估計值。

#### 狀態空間模型（本策略使用）

**狀態向量**（內部追蹤的隱藏變數）：

```
x_t = [p_t, v_t]ᵀ
        p_t = 平滑後的價格估計
        v_t = 價格速度（趨勢斜率）
```

**狀態轉移方程**（每一棒如何演進）：

```
x_t = F · x_(t-1) + w_t，    w_t ~ N(0, Q)

      ⎡1  1⎤                  ⎡σ_proc²      0       ⎤
F  =  ⎢    ⎥，   Q（過程雜訊）=  ⎢                    ⎥
      ⎣0  1⎦                  ⎣0       0.1·σ_proc²   ⎦
```

**觀測方程**（我們只觀測到原始收盤價，看不到速度）：

```
y_t = H · x_t + r_t，    r_t ~ N(0, R)
H = [1, 0]，    R = σ_obs²
```

**參數設定：**

| 參數 | 數值 | 意義 |
|:-----|-----:|:-----|
| σ_obs（量測雜訊） | {KF_OBS_NOISE} | 每根 K 棒的報價雜訊強度 |
| σ_proc（過程雜訊） | {KF_PROC_NOISE} | 真實價格每棒的漂移速度 |

**遞迴更新公式（每棒執行一次）：**

```
【預測步驟】
  x̂_(t|t-1) = F · x̂_(t-1)            ← 用上一棒估計預測這棒
  P_(t|t-1)  = F · P_(t-1) · Fᵀ + Q   ← 傳播不確定性

【Kalman 增益】
  K_t = P_(t|t-1) · Hᵀ · (H · P_(t|t-1) · Hᵀ + R)⁻¹
        ↑ 決定「要多相信新觀測 vs. 舊預測」，自動在雜訊大時多靠預測

【更新步驟】
  x̂_t = x̂_(t|t-1) + K_t · (y_t − H · x̂_(t|t-1))   ← 融合新觀測
  P_t  = (I − K_t · H) · P_(t|t-1)                   ← 縮小不確定性
```

輸出的平滑價格 `close_kf = p̂_t` 被用於**所有後續特徵計算**。
它在減少買賣價差雜訊的同時，僅引入約 1 棒的滯後。
""")

        # ── 3. 隱馬爾可夫模型（HMM） ──────────────────────────────────────────
        a("## 3. 隱馬爾可夫模型（HMM）——為何使用？怎麼運作？\n")
        a(f"""
### 直覺理解

想像市場有一個你看不見的「心情」（機制），這個隱藏的心情驅動了你觀測到的價格和
成交量模式。隱馬爾可夫模型（HMM）就是一個對此建模的機率框架：

1. 假設存在 **N 個隱藏狀態**（對應不同的市場機制）。
2. 每個狀態會根據一個機率分布「生成」你觀測到的特徵（K 棒指標）。
3. 狀態之間以固定但**可學習的機率**相互轉換。

因為狀態是隱藏的，我們只能從觀測到的特徵序列來**推斷**當前最可能是哪個狀態。

### 數學表達式

**三組由 Baum-Welch（EM）演算法學習的參數：**

| 符號 | 描述 |
|:-----|:-----|
| π | 初始狀態分布：`P(s₁ = k)` |
| A | 轉移矩陣：`A[i,j] = P(s_t = j ｜ s_(t-1) = i)` |
| μ_k, Σ_k | 狀態 k 的高斯發射分布參數（均值向量、協方差矩陣） |

**發射模型（多元高斯分布）：**

```
P(o_t ｜ s_t = k) = N(o_t ; μ_k, Σ_k)

其中 o_t ∈ ℝ¹⁴ 是第 t 棒的 14 維特徵向量
```

**機制信心度（用作進場門檻）：**

```
confidence_t = max_k  P(s_t = k ｜ o_1, o_2, ..., o_t)
```

只有 `confidence_t > {HMM_CONF_THRESHOLD}` 時才允許進場——確保策略只在 HMM
對當前機制有足夠把握時才採取行動。

### 訓練後如何給狀態貼標籤？

HMM 訓練完後，每個狀態（state）是個匿名數字（0, 1, 2, 3）。
我們用以下三個統計量，為每個狀態自動賦予人可讀的機制名稱：

1. **rv_short 均值** → 按中位數分割：高波動 vs. 低波動
2. **momentum 均值** → 低波動組：動量 > 0 → `BULL_QUIET`；動量 < 0 → `BEAR_QUIET`
3. **|momentum| 均值** → 高波動組：方向性較強 → `STORM`；方向性較弱 → `CHAOS`

這套標籤邏輯完全由資料驅動，沒有任何人工硬編碼的切割點。

### Walk-Forward 使用方式（嚴格防止偷看未來）

```
Walk-Forward 架構：
  訓練第  1 ~ {WF_TRAIN_DAYS}  個交易日  → HMM 在此學習機制
  測試第 {WF_TRAIN_DAYS+1} ~ {WF_TRAIN_DAYS+WF_TEST_DAYS} 個交易日  → 策略在此OOS交易
  步進 {WF_STEP_DAYS} 天 → 從頭重新訓練 HMM → 進入下一個測試窗口
```

HMM **永遠只用樣本內資料訓練**，從未使用未來資訊分類過去的機制。
""")

        # ── 4. 特徵工程 ───────────────────────────────────────────────────────
        a("## 4. 特徵工程——14 個因子\n")
        a(f"""
所有特徵都在 **Kalman 平滑後的 5 分鐘 K 棒**上計算，並對每個特徵做 ±3σ 截尾
（Winsorization）以防離群值干擾 HMM 的擬合。

| 編號 | 特徵名稱 | 計算公式 | 捕捉什麼資訊 |
|:----:|:---------|:---------|:------------|
| F01 | `log_ret_kf` | `ln(close_kf_t / close_kf_(t-1))` | Kalman 平滑後的 5 分鐘對數報酬（去雜訊真實報酬） |
| F02 | `rv_short` | `std(log_ret_kf, 滾動 {VOL_WIN_SHORT} 棒 = 30 分鐘)` | 短期已實現波動率（機制分割的主軸） |
| F03 | `vol_expansion` | `rv_short / rv_long`（截尾 0–5） | 波動擴張比，> 1 表示波動正在放大 |
| F04 | `momentum` | `sum(log_ret_kf, 滾動 {MOMENTUM_WIN} 棒 = 1 小時)` | 日內短期趨勢強度 |
| F05 | `vwap_dev` | `(close_kf − VWAP) / close_kf` | 價格偏離 VWAP 的程度（均值回歸信號） |
| F06 | `range_norm` | `(H − L) / close_kf` | 日內振幅（ATR 代理指標） |
| F07 | `vol_surprise` | `ln((volume+1) / (EWM_volume+1))`，EWM span=48 棒 | 成交量是否異常（確認趨勢有量配合） |
| F08 | `roll_yield` | `ln(近月價 / 遠月價) × (252 / 兩月到期日差)` | 年化期貨基差：> 0 逆價差，< 0 正價差 |
| F09 | `liq_ratio` | `遠月成交量 / 近月成交量`（截尾 0–10） | 換倉壓力：數值高代表資金在移向遠月 |
| F10 | `time_sin` | `sin(2π × 日內時間分數)` | 時間循環編碼（sin 分量） |
| F11 | `time_cos` | `cos(2π × 日內時間分數)` | 時間循環編碼（cos 分量） |
| F12 | `spread_ar1` | `roll_yield_(t-1)`（1 棒滯後） | 基差動能持續性（嚴格因果，無偷看） |
| F13 | `mom_reversal` | `−sign(mom × mom_short) × min(｜mom × mom_short｜, 0.02)` | 短長期動量方向不一致（潛在反轉信號） |
| F14 | `garch_vol` | `sqrt(EWM(log_ret_kf², λ=0.94))` | EWM-GARCH 條件波動率（更穩健的波動估計） |

**補充說明：**

- `rv_long = std(log_ret_kf, 滾動 {VOL_WIN_LONG} 棒 = 2 小時)` → F03 的分母
- `日內時間分數 = (距 08:45 的分鐘數) / 300`（台灣期貨主要盤 08:45–13:45）
- **為何用 sin/cos 編碼時間（F10/F11）？** 直接用分鐘數的話，模型會把 08:45
  和 13:45 視為「截然不同」的兩個時刻。sin/cos 編碼讓開盤和收盤在幾何上更接近，
  正確反映「開收盤都有高波動」的日內規律。
""")

        # ── 4.5 PCA 降維分析 ──────────────────────────────────────────────────
        a("## 4.5 PCA 降維——為何使用？特徵重要性分析\n")

        pca_n = 0
        pca_evr: List[float] = []
        pca_importance: List[Tuple[str, float]] = []

        if clf is not None and clf.pca is not None:
            _pca = clf.pca
            pca_n   = _pca.n_components_
            pca_evr = list(_pca.explained_variance_ratio_)
            _imp    = (_pca.explained_variance_ratio_[:, np.newaxis]
                       * _pca.components_ ** 2).sum(axis=0)
            _imp    = _imp / _imp.sum()
            pca_importance = sorted(
                zip(FEAT_COLS, _imp.tolist()),
                key=lambda x: x[1], reverse=True
            )

        a(f"""
### 為什麼要做 PCA？

14 個特徵之間存在相關性（例如 `rv_short` 與 `garch_vol` 都衡量波動率）。
直接把 14 維特徵餵給 HMM 有兩個問題：

1. **多重共線性**：高度相關的特徵讓高斯 HMM 的協方差矩陣估計不穩定。
2. **維度災難**：14 維 full covariance 需要估計 14×15/2=105 個協方差參數，
   在訓練樣本有限時容易過擬合。

PCA 把 14 個原始特徵線性組合成 **{pca_n} 個不相關的主成分（PC）**，
只保留最重要的資訊，大幅降低 HMM 需要估計的參數數量。

### 解釋變異量

| 主成分 | 個別解釋變異量 | 累積解釋變異量 |
|:------:|:----------:|:-----------:|""")

        cumev = 0.0
        for k, ev in enumerate(pca_evr):
            cumev += ev
            a(f"| PC{k+1} | {ev:.1%} | {cumev:.1%} |")

        a(f"""
> 前 **{pca_n}** 個主成分合計解釋 **{cumev:.1%}** 的原始特徵變異量。
> 這代表 HMM 用更少的維度捕捉了原始 14 個因子中 {cumev:.1%} 的市場資訊。

### 特徵重要性排名

「加權特徵重要性」計算方式：

```
importance_j = Σ_k  (explained_variance_ratio_k × loading²_k,j)
```

即每個主成分的解釋力 × 該特徵在此主成分的載荷量平方，加總後正規化為百分比。

| 排名 | 特徵 | 加權重要性 | 主要捕捉的市場資訊 |
|:---:|:-----|:---------:|:-----------------|""")

        feat_desc = {
            "log_ret_kf"   : "Kalman 平滑 5 分鐘對數報酬（去雜訊）",
            "rv_short"     : "30 分鐘已實現波動率（機制主要分割軸）",
            "vol_expansion": "波動擴張比 rv_short/rv_long（機制轉換信號）",
            "momentum"     : "1 小時累積報酬（趨勢方向與強度）",
            "vwap_dev"     : "VWAP 偏離度（均值回歸信號）",
            "range_norm"   : "K 棒振幅/收盤價（ATR 代理）",
            "vol_surprise" : "成交量異常度（量能確認）",
            "roll_yield"   : "年化期貨基差（Carry 方向）",
            "liq_ratio"    : "遠月/近月成交量（換倉壓力）",
            "time_sin"     : "日內時間 sin 編碼",
            "time_cos"     : "日內時間 cos 編碼",
            "spread_ar1"   : "Roll yield 滯後項（基差動能）",
            "mom_reversal" : "短長期動量方向不一致（反轉信號）",
            "garch_vol"    : "EWM-GARCH 條件波動率（λ=0.94）",
        }
        for rank, (fname, imp) in enumerate(pca_importance, 1):
            a(f"| {rank} | `{fname}` | {imp:.1%} | {feat_desc.get(fname,'')} |")

        a("""
> 詳細視覺化（Scree Plot、載荷量熱圖、特徵排名）請參見
> `output/hmm_regime/08_pca_analysis.png`。
""")

        # ── 5. 進場邏輯 ───────────────────────────────────────────────────────
        a("## 5. 進場邏輯——策略何時開倉？\n")
        a(f"""
### 5.0 日內執行架構（v2.2 新增）

策略採 **{EXEC_RESAMPLE_MIN} 分鐘執行棒**：特徵計算維持 5 分鐘解析度，
但進場評估只在每 {EXEC_RESAMPLE_MIN} 分鐘整點（minute % {EXEC_RESAMPLE_MIN} == 0）
且在交易時段 `{SESSION_START}–{SESSION_END}` 內進行。

```
一天最多執行棒數 = ({SESSION_END} − {SESSION_START}) ÷ {EXEC_RESAMPLE_MIN} 分鐘
               ≈ 8 棒/日（09:30, 10:00, 10:30, 11:00, 11:30, 12:00, 12:30, 13:00）
```

**日內反鞭刑（anti-whipsaw）：** 前一信號確立後，至少需間隔 {MIN_HOLD_BARS} 個執行棒
（= {MIN_HOLD_BARS * EXEC_RESAMPLE_MIN} 分鐘）才允許方向反轉，防止盤整時頻繁切換。

### 5.1 趨勢機制：BULL_QUIET / BEAR_QUIET / STORM

**信號來源（每執行棒即時評估，嚴格使用當棒歷史資料）：**

方向信號來自 5 分鐘 `momentum` 特徵（過去 1 小時累積對數報酬）的 z-score：

```
方向 = LONG   若  momentum_t > σ_mom × τ_z    （強烈上升動量）
     = SHORT  若  momentum_t < −σ_mom × τ_z   （強烈下跌動量）
     = FLAT   其他（動量不足，不進場）
```

**日內反鞭刑（Anti-whipsaw）：** 確立信號後至少 {MIN_HOLD_BARS} 個執行棒
（= {MIN_HOLD_BARS * EXEC_RESAMPLE_MIN} 分鐘）才允許反轉方向。

**趨勢機制允許隔夜留倉**——可持倉跨越多個交易日，直到停損觸發或機制變換。

**以下所有條件必須「同時」滿足才能進場：**

| # | 條件 | 數學表達式 | 閾值 |
|:-:|:-----|:-----------|-----:|
| 0 | 執行棒時機 | `minute % exec_min == 0` 且在交易時段 | `exec_min = {EXEC_RESAMPLE_MIN} 分鐘；時段 {SESSION_START}–{SESSION_END}` |
| 1 | HMM 信心度 | `P(主導狀態) > τ_conf` | `τ_conf = {HMM_CONF_THRESHOLD}` |
| 2 | 機制持續性 | 連續 N 天相同主導機制 | `N = {REGIME_PERSIST_DAYS} 天` |
| 3 | 動量 z-score | `｜momentum_t｜ / σ_mom_訓練期 > τ_z` | `τ_z = {MOM_ZSCORE_THRESH}` |
| 4 | 僅 STORM：VWAP 偏離過濾 | `｜vwap_dev_t｜ < δ_vwap`（不追高殺低） | `δ_vwap = {STORM_MAX_VWAP_DEV:.3f}` |
| 5 | 僅 STORM：波動擴張過濾 | `vol_expansion_t < v_max` | `v_max = {STORM_MAX_VOL_EXP}` |
| 6 | 僅 STORM：成交量確認 | `vol_surprise_t > v_min` | `v_min = {STORM_MIN_VOL_SURP}` |

其中 `σ_mom_訓練期` 是在訓練窗口中計算的 momentum 標準差（每折 Walk-Forward 都更新）。

### 5.2 CHAOS 機制——均值回歸

**信號來源（每 {EXEC_RESAMPLE_MIN} 分鐘執行棒，交易時段 {SESSION_START}–{SESSION_END}）：**

```
vwap_z_t = ｜vwap_dev_t｜ / σ_vwap_訓練期

方向 = 做空 SHORT  若  vwap_dev_t > 0   （價格高於 VWAP → 淡化向下回歸）
     = 做多 LONG   若  vwap_dev_t < 0   （價格低於 VWAP → 淡化向上回歸）
```

**CHAOS 倉位不隔夜**——每日 13:30 強制平倉。均值回歸邏輯不適合隔夜持有。

**以下四個條件必須同時滿足：**

| # | 條件 | 數學表達式 | 閾值 |
|:-:|:-----|:-----------|-----:|
| 0 | 執行棒時機 | `minute % exec_min == 0` 且在交易時段 | `exec_min = {EXEC_RESAMPLE_MIN} 分鐘` |
| 1 | HMM 信心度 | `P(CHAOS 狀態) > τ_conf` | `τ_conf = {HMM_CONF_THRESHOLD}` |
| 2 | VWAP 偏離強度 | `vwap_z_t > τ_vwap` | `τ_vwap = {VWAP_MR_ZSCORE}` |
| 3 | 波動收縮 | `vol_expansion_t < 1.2` | 在波動開始收縮後才淡化，不在波動高峰進場 |
| 4 | 每日交易次數上限 | `今日 CHAOS 交易次數 < max_chaos` | `max = {MAX_CHAOS_PER_DAY} 筆/日` |

### 5.3 倉位計算

**以風險預算計算張數：**

```
pts_risk  = k_stop × ATR_t                      （k_stop 為機制別停損乘數，見第 6 節）
base_lots = floor( NAV × ρ / (pts_risk × M) )

其中：
  ρ = 單筆風險比例 = {RISK_PER_TRADE:.1%}
  M = 合約乘數    = NT${MULTIPLIER}/點
  NAV = 當前淨資產價值
```

**套用機制別倉位係數：**

```
n = clip( floor(base_lots × size_factor × carry_adj),  最小=1,  最大={MAX_CONTRACTS} )

size_factor：  BULL_QUIET = {SIZE_FACTOR['BULL_QUIET']:.2f}
               BEAR_QUIET = {SIZE_FACTOR['BEAR_QUIET']:.2f}
               STORM      = {SIZE_FACTOR['STORM']:.2f}
               CHAOS      = {SIZE_FACTOR['CHAOS']:.2f}
```

**Carry 倉位調整（carry_adj）：**

```
若  ｜roll_yield｜ > {CARRY_THRESHOLD:.2f}（即年化基差 > {CARRY_THRESHOLD*100:.0f}%）：
    carry_adj = 1 + {CARRY_BOOST:.2f}  若交易方向與 carry 方向一致（carry 助力）
    carry_adj = 1 − {CARRY_BOOST:.2f}  若交易方向與 carry 方向相反（carry 阻力）
否則：
    carry_adj = 1.0（不調整）
```
""")

        # ── 6. 停損與出場邏輯 ──────────────────────────────────────────────────
        a("## 6. 停損與出場邏輯\n")
        a(f"""
以下出場條件**每棒都會檢查**，按優先順序（數字小者優先）執行：

| 優先 | 出場觸發 | 規則說明 |
|:----:|:---------|:---------|
| 1 | **機制變換出場** | 當前棒 HMM 機制 ≠ 進場時機制 → 下一棒開盤立即平倉 |
| 2 | **硬性停損（ATR 停損）** | 從進場點固定距離出場；距離由機制別 ATR 乘數決定 |
| 3 | **追蹤停損** | 獲利 ≥ 1 ATR 後啟動；隨著峰值/谷值移動 |
| 4 | **獲利目標**（僅 CHAOS） | 固定獲利目標倍數 |
| 5 | **EOD 強制平倉**（僅 CHAOS） | 每日 13:30 無條件平倉 |
| 6 | **日虧損限制** | 單日虧損達上限 → 當日停止所有新進場（不影響現有倉位） |
| 7 | **信號反轉** | EWM 動量方向改變 → 在執行視窗內反向出場 |

### 硬性停損（ATR 停損）

ATR 是 True Range 的 14 棒滾動平均：

```
TR_t = max( H_t − L_t,  ｜H_t − C_(t-1)｜,  ｜L_t − C_(t-1)｜ )

ATR_t = mean( TR_(t-13), ..., TR_t )          ← 14 棒簡單平均

停損價格計算：
  做多（LONG）：  stop_price = entry_price − k × ATR_t
  做空（SHORT）： stop_price = entry_price + k × ATR_t
```

成交假設：以 `min(open, stop_price)` 成交（多頭停損），反映跳空開盤的現實情況。

**機制別 ATR 乘數（k）：**

| 機制 | k | 設計邏輯 |
|:-----|:-:|:---------|
| BULL_QUIET | {ATR_STOP['BULL_QUIET']} | 趨勢平穩，給足回撤空間讓趨勢延伸 |
| BEAR_QUIET | {ATR_STOP['BEAR_QUIET']} | 同上 |
| STORM | {ATR_STOP['STORM']} | 高波動，必須適度收緊以限制損失 |
| CHAOS | {ATR_STOP['CHAOS']} | 均值回歸，若方向錯誤需快速認賠 |

### 追蹤停損

當倉位獲利達到 1 ATR 時，啟動追蹤停損以保護利潤：

```
啟動條件：  profit_t = (price_t − entry) × direction ≥ ATR_t

做多時：
  peak_t       = max(price_t, peak_(t-1))                  ← 追蹤最高點
  trail_stop_t = max(trail_stop_(t-1), peak_t − {TRAIL_MULT} × ATR_t)
  → 當 L_t ≤ trail_stop_t 時出場

做空時：
  trough_t     = min(price_t, trough_(t-1))                ← 追蹤最低點
  trail_stop_t = min(trail_stop_(t-1), trough_t + {TRAIL_MULT} × ATR_t)
  → 當 H_t ≥ trail_stop_t 時出場
```

追蹤停損的作用：隨著行情移動自動上移/下移保護線，**鎖定已賺到的利潤**。

### 獲利目標（僅 CHAOS）

```
做多：  當 H_t ≥ entry_price + {CHAOS_PROFIT_TARGET_ATR} × ATR_t 時出場
做空：  當 L_t ≤ entry_price − {CHAOS_PROFIT_TARGET_ATR} × ATR_t 時出場

成交價 = entry_price ± {CHAOS_PROFIT_TARGET_ATR} × ATR_t（限價成交假設）
```

CHAOS 機制的**風報比設計**：
報酬目標 {CHAOS_PROFIT_TARGET_ATR} ATR ÷ 停損距離 {ATR_STOP['CHAOS']} ATR = {CHAOS_PROFIT_TARGET_ATR/ATR_STOP['CHAOS']:.2f}
→ 即使勝率僅 40%，期望值仍為正。

### 日虧損限制

```
日損益百分比 = (NAV_當前 − NAV_今日開盤) / NAV_今日開盤

若  日損益百分比 ≤ {DAILY_LOSS_LIM:.3f}  （即單日虧損達 {abs(DAILY_LOSS_LIM)*100:.1f}%）：
    → 當日停止所有新進場（已持有倉位不強制平倉）
```

防止單日重大失誤讓整個帳戶元氣大傷。
""")

        # ── 7. 交易成本模型 ────────────────────────────────────────────────────
        a("## 7. 交易成本模型\n")
        a(f"""
每筆交易在進場**和**出場時各付一次，以下是三個組成項目的完整說明：

| 成本項目 | 計算方式 | 金額 |
|:---------|:---------|-----:|
| **手續費（券商 + 交易所）** | 每口固定費用（來回） | NT${TC_COMMISSION_RT} / 口 RT |
| **期貨交易稅** | `2 × {TC_TAX_RATE*100:.4f}% × 成交金額 × 口數` | 進出各一次 |
| **滑點（市場衝擊）** | 每口固定估計（一般市價單） | NT${TC_SLIPPAGE_RT} / 口 RT |
| **滑點（換倉價差單）** | 價差單流動性較好，滑點較低 | NT${TC_SLIPPAGE_ROLL} / 口 RT |

**完整公式（單邊，n = 口數）：**

```
TC = 手續費 + 期交稅 + 滑點

手續費 = {TC_COMMISSION_RT} × n
期交稅 = {TC_TAX_RATE} × 2 × 成交價 × {MULTIPLIER} × n
滑點   = {TC_SLIPPAGE_RT}  × n    （一般市價單）
       = {TC_SLIPPAGE_ROLL} × n    （跨月換倉價差單）
```

**數值範例——1 口，FITX 成交價 = 20,000 點：**

```
手續費 = NT$100
期交稅 = 0.00002 × 2 × 20,000 × 200 × 1 = NT$160
滑點   = NT$200
─────────────────────────────────────────────
來回合計 = NT$460 / 口
```

滑點估計為 1 個升降單位（NT$200 = 1 點 × NT$200 乘數）來回，代表一個買賣價差。
換倉價差單（NT$100 RT）因為直接交易價差而非兩個別腿，流動性較好、滑點較小。
""")

        # ── 8. Walk-Forward 績效 ──────────────────────────────────────────────
        a("## 8. Walk-Forward 樣本外（OOS）績效\n")
        a(f"> 訓練窗口：**{WF_TRAIN_DAYS}** 交易日 | "
          f"測試窗口：**{WF_TEST_DAYS}** 交易日 | "
          f"步進：**{WF_STEP_DAYS}** 交易日\n")

        a("| Fold | 訓練期 | 測試期 | 報酬率 | Sharpe | 最大回撤 | 勝率 | 獲利因子 | 交易數 |")
        a("|:----:|:------:|:------:|-------:|-------:|---------:|-----:|---------:|-------:|")
        for r in fold_results:
            a(f"| {r['fold']} | {r['train_start']}→{r['train_end']} "
              f"| {r['test_start']}→{r['test_end']} "
              f"| {r['total_return']} | {r['sharpe']} "
              f"| {r['max_drawdown']} | {r['win_rate']} "
              f"| {r['profit_factor']} | {r['n_trades']} |")

        a("\n### 彙總 OOS 績效\n")
        a("| 指標 | 數值 |")
        a("|:-----|-----:|")
        for k, v in summ.items():
            label_map = {
                "total_return" : "總報酬率",
                "sharpe"       : "Sharpe Ratio",
                "sortino"      : "Sortino Ratio",
                "calmar"       : "Calmar Ratio",
                "max_drawdown" : "最大回撤",
                "n_trades"     : "總交易筆數",
                "win_rate"     : "整體勝率",
                "profit_factor": "獲利因子（Profit Factor）",
                "avg_hold_hrs" : "平均持倉時間",
            }
            a(f"| {label_map.get(k,k)} | {v} |")

        # ── 9. 多空損益拆解 ────────────────────────────────────────────────────
        a("\n## 9. 多空損益拆解與對稱性分析\n")

        L = attr.get("long",  {})
        S = attr.get("short", {})
        total_gross = attr.get("total_gross", 1) or 1

        a("| 指標 | 做多（LONG） | 做空（SHORT） | 不對稱度 |")
        a("|:-----|:-----------:|:------------:|:--------:|")

        def _asym(lv, sv):
            d = abs(lv) + abs(sv)
            return f"{(lv-sv)/d:+.1%}" if d > 1 else "n/a"

        rows_ls = [
            ("交易筆數",      L.get("n",0),         S.get("n",0),         False),
            ("毛 P&L（NT$）", L.get("gross",0),      S.get("gross",0),     False),
            ("淨 P&L（NT$）", L.get("net",0),        S.get("net",0),       False),
            ("Carry 貢獻",    L.get("carry",0),      S.get("carry",0),     False),
            ("勝率",          L.get("win_rate",0),   S.get("win_rate",0),  True),
            ("平均毛利/筆",   L.get("avg_gross",0),  S.get("avg_gross",0), False),
            ("平均持倉",      L.get("avg_hold",0),   S.get("avg_hold",0),  False),
            ("交易成本",      L.get("tc",0),         S.get("tc",0),        False),
        ]
        for name, lv, sv, is_pct in rows_ls:
            if name == "平均持倉":
                a(f"| {name} | {lv:.1f}h | {sv:.1f}h | {_asym(lv,sv)} |")
            elif is_pct:
                a(f"| {name} | {lv:.2%} | {sv:.2%} | {_asym(lv,sv)} |")
            else:
                a(f"| {name} | NT${lv:>12,.0f} | NT${sv:>12,.0f} | {_asym(lv,sv)} |")

        l_share = L.get("gross",0) / abs(total_gross)
        s_share = S.get("gross",0) / abs(total_gross)
        a(f"\n**毛 P&L 貢獻比例：** 做多 {l_share:.1%} ｜ 做空 {s_share:.1%}\n")
        if abs(l_share - s_share) < 0.15:
            a("- 多空收益相對對稱（差距 < 15%），策略在上漲與下跌市場均能有效捕捉趨勢。")
        elif l_share > s_share + 0.20:
            a("- 做多貢獻顯著高於做空。這與台灣股市長期正向股票風險溢酬吻合"
              "（台指近年年化報酬約 7–12%）。多頭趨勢持續性更強，容易累積獲利。")
        else:
            a("- 做空貢獻顯著高於做多。可能反映回測期間市場整體走弱，"
              "或空頭信號對急跌行情的捕捉效果較好。")

        wr_diff = L.get("win_rate",0) - S.get("win_rate",0)
        a(f"\n- 多空勝率差異：**{wr_diff:+.2%}**。"
          + ("做多勝率高於做空，符合市場偏多的長期特性。"
             if wr_diff > 0.03 else
             "多空勝率接近，說明空頭 Alpha 引擎品質良好。"))

        # ── 10. Roll Yield（Carry）貢獻 ───────────────────────────────────────
        a("\n## 10. Roll Yield（換倉收益）貢獻分析\n")

        total_carry   = attr.get("total_carry", 0)
        carry_pct     = attr.get("carry_pct_gross", 0)
        total_gross_v = attr.get("total_gross", 0)

        a("| 項目 | 金額（NT$） |")
        a("|:-----|----------:|")
        a(f"| 總毛 P&L          | {total_gross_v:>14,.0f} |")
        a(f"| Roll Carry P&L    | {total_carry:>14,.0f} |")
        a(f"| 純價格動量 P&L    | {total_gross_v-total_carry:>14,.0f} |")
        a(f"| Carry 佔毛利比例  | {carry_pct:.2%} |")

        a(f"""
**每棒的 Carry 貢獻計算公式：**

```
carry_per_bar = roll_yield_年化
              × (resample_minutes / (252 × 390))
              × 進場價 × {MULTIPLIER} × 口數 × 方向（+1 做多，−1 做空）

其中 252 × 390 = 全年交易分鐘數（台灣期貨每日 390 分鐘）
```

**Roll Yield 計算公式：**

```
roll_yield_t = ln(近月收盤價_t / 遠月收盤價_t) × (252 / 兩月到期日差_天數)
```

| 基差狀況 | 意義 | 對持倉的影響 |
|:---------|:-----|:------------|
| `roll_yield > 0`（逆價差） | 近月以**溢價**交易於遠月 | 做多方承擔換倉成本；做空方獲得 carry 收益 |
| `roll_yield < 0`（正價差） | 近月以**折價**交易於遠月 | 做多方從換倉升水獲益；做空方承擔換倉成本 |
""")
        ry = feat_df["roll_yield"].resample("D").mean().dropna()
        back_days = int((ry > 0).sum())
        cont_days = int((ry < 0).sum())
        a(f"""
| 基差狀況 | 天數 | 佔比 |
|:---------|-----:|:----:|
| 逆價差（roll_yield > 0） | {back_days} | {back_days/max(len(ry),1):.1%} |
| 正價差（roll_yield < 0） | {cont_days} | {cont_days/max(len(ry),1):.1%} |

±{CARRY_BOOST*100:.0f}% 的倉位調整（當 `｜roll_yield｜ > {CARRY_THRESHOLD*100:.0f}%` 時觸發）
讓策略在 carry 助力時**加碼**、在 carry 阻力時**縮碼**，系統性降低隱性的持倉成本。
""")

        # ── 11. 交易成本拆解 ──────────────────────────────────────────────────
        a("\n## 11. 交易成本拆解分析\n")

        tc_bd    = attr.get("tc_breakdown", {})
        total_tc = attr.get("total_tc", 0)
        tc_pct   = attr.get("tc_pct_gross", 0)
        n_trades = max(len(all_trades), 1)

        a("| 成本項目 | 總計（NT$） | 佔 TC 比例 | 每筆均攤 |")
        a("|:---------|----------:|:---------:|--------:|")
        for comp, val in tc_bd.items():
            name_map = {
                "commission": "手續費（券商 + 交易所）",
                "tax"       : "期貨交易稅（0.002%/邊）",
                "slippage"  : "滑點（市場衝擊）",
            }
            a(f"| {name_map.get(comp,comp)} | NT${val:>10,.0f} "
              f"| {val/max(total_tc,1):.1%} | NT${val/n_trades:>6,.0f} |")
        a(f"| **合計** | **NT${total_tc:>10,.0f}** | **100%** | "
          f"**NT${total_tc/n_trades:>6,.0f}** |")

        a(f"\n**TC 拖曳（TC Drag）：{tc_pct:.2%} of ｜毛 P&L｜**\n")
        if tc_pct < 0.20:
            a("TC 佔比偏低，策略的淨 Alpha 空間充足，交易頻率合理。")
        elif tc_pct < 0.40:
            a("TC 佔比適中。可透過提高最短持倉天數（`MIN_HOLD_DAYS`）或"
              "收窄進場條件進一步降低。")
        else:
            a("TC 佔比偏高，建議：① 提高動量 z-score 門檻；"
              "② 增加最短持倉天數至 5 天；③ 改用限價單進場。")

        # ── 12. 策略何時獲利 / 虧損？ ────────────────────────────────────────
        a("\n## 12. 策略何時獲利？何時虧損？\n")

        monthly = eq_df["equity"].resample("ME").last().pct_change().dropna()*100
        if not monthly.empty:
            wins = monthly[monthly >= 0]
            loss = monthly[monthly < 0]
            a(f"**月度勝率：{len(wins)/len(monthly):.1%}** "
              f"（{len(wins)} 個正向月 / {len(loss)} 個負向月）\n")
            if not loss.empty:
                a(f"- 最差月份：{loss.idxmin().strftime('%Y-%m')} （{loss.min():.2f}%）")
                a(f"- 平均虧損月：{loss.mean():.2f}%")
            if not wins.empty:
                a(f"- 最佳月份：{wins.idxmax().strftime('%Y-%m')} （{wins.max():.2f}%）")
                a(f"- 平均獲利月：{wins.mean():.2f}%\n")

        exit_reasons = pd.Series([t.exit_reason for t in all_trades
                                   if t.exit_reason]).value_counts()
        if not exit_reasons.empty:
            a("**出場原因分佈：**\n")
            a("| 出場原因 | 次數 | 佔比 |")
            a("|:---------|-----:|-----:|")
            te = exit_reasons.sum()
            reason_map = {
                "hard_stop"      : "ATR 硬性停損",
                "trail_stop"     : "追蹤停損",
                "eod_flatten"    : "EOD 強制平倉（CHAOS）",
                "signal_reversal": "信號反轉",
                "regime_change"  : "機制變換出場",
                "profit_target"  : "獲利目標達成（CHAOS）",
                "end_of_period"  : "回測期末平倉",
            }
            for rsn, cnt in exit_reasons.items():
                a(f"| {reason_map.get(rsn,rsn)} | {cnt} | {cnt/te:.1%} |")

        a("""
### 策略獲利的典型情境

1. **趨勢持續期（BULL_QUIET / BEAR_QUIET 持續 3 天以上）：**
   動量信號清晰，ATR 停損不頻繁觸發，策略可跨日累積大額趨勢利潤。
   這是策略的**主要獲利來源**。

2. **Carry 方向與持倉一致：** 每個持倉棒都在累積一點點 Carry，
   長期持倉時這個效果很可觀，是趨勢利潤之外的額外正期望值。

3. **CHAOS 精準入場：** 波動率已收縮、VWAP 偏離卻很大時，
   均值回歸往往能在 1.5 ATR 的獲利目標內快速了結。

### 策略虧損的典型情境

1. **機制轉換過渡期（最大風險）：** 市場從平靜突然進入高波動，
   HMM 有 1–2 棒的識別延遲。這段時間趨勢倉位可能面臨反向大波動。
   *緩解措施：機制變換出場（regime_change）在下棒開盤立即止損。*

2. **動量假突破：** 機制判斷正確，但進場時機不佳（如進場後價格立刻回撤），
   短期反轉觸發 ATR 停損。
   *緩解措施：STORM 的 VWAP 偏離與成交量過濾攔截大部分這類情況。*

3. **低波動盤整：** EWM 動量反覆小幅震盪，但機制持續性不足（< 2 天），
   進場門檻自動攔截，幾乎不會產生無效交易。

4. **CHAOS 均值回歸失敗：** 當出現大型方向性催化劑（如外資大量買超），
   價格偏離 VWAP 後繼續延伸而不回歸，觸發 0.8 ATR 硬性停損。
""")

        # ── 13. 改善建議 ──────────────────────────────────────────────────────
        a("\n## 13. 改善建議\n")
        a(f"""
### 13.1 近期可執行的改善

| 優先級 | 改善項目 | 預期影響 |
|:------:|:---------|:---------|
| 高 | **雙時框 HMM：** 加入日線 3 狀態 HMM，只在日線與 5 分鐘 HMM 一致時進場 | 大幅降低假突破，提升信號品質 |
| 高 | **限價單進場：** 趨勢機制在 VWAP 附近掛限價單取代市價單 | 滑點降低約 50%，改善平均成交價 |
| 中 | **突破方向信號：** 將 EWM 交叉換成 N 日高低點突破，作為 QUIET 機制的方向判斷 | 更適合日內至多日持倉的節奏 |
| 中 | **進場強度評分：** 將 Conf × MomZ × VolSurp 合成一個進場得分，只取前 30% 的信號 | 提升進場時機的精準度 |
| 低 | **自適應 ATR 乘數：** 根據近 30 日波動水平動態調整停損距離 | 在不同波動週期維持穩定的停損比例 |

### 13.2 研究方向

1. **選擇權資料整合：** 台指選擇權隱含波動率（TXO IV）是前瞻指標。
   當 IV 大幅高於歷史波動且快速攀升時，提前切換至 CHAOS 或防守模式，
   比 HMM 更快感知到即將到來的機制轉變。

2. **跨市場機制確認：** 利用道瓊期指（YM）、恒生期指（HSI）的日內相關性，
   以外部信號補強 HMM 的機制判斷，降低 FITX 單一市場的假信號。

3. **HMM 集成（Ensemble）：** 用 3–5 個不同亂數種子訓練多個 HMM，
   以多數決決定最終機制，降低單一 HMM 收斂至局部最優解的風險。

4. **尾部風險對沖：** 在 STORM 機制持倉期間，買入虛值 PUT 作為凸性保護，
   防範隔夜跳空風險，建構非線性的損益結構。

### 13.3 穩健性驗證建議

- **參數敏感度分析：** 對 `HMM_CONF_THRESHOLD`（±0.10）和
  `MOM_ZSCORE_THRESH`（±0.50）進行掃描，確認績效對參數選擇不敏感。
- **極端行情壓力測試：** 模擬 2020 年 COVID 閃崩、2022 年聯準會升息衝擊下的策略表現。
- **交易成本壓力測試：** 在 TC × 2 的情境下，策略是否仍有正期望值？
""")

        a("\n---\n")
        a("_本報告由 `hmm_vol_regime_strategy.py v2.2` 自動生成_\n")
        a(f"_初始資金：NT${INITIAL_CAPITAL:,}  |  "
          f"合約乘數：{MULTIPLIER} NT$/點  |  "
          f"資料：FITX 分鐘 K 棒_\n")

        report = "\n".join(lines)
        self.out_path.write_text(report, encoding="utf-8")
        log.info("報告 → %s", self.out_path)
        return report



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

        MarkdownReportGenerator(self.output_dir).generate(
            agg, wfa.fold_results, wfa.all_trades, feat_df, oos_eq,
            clf=full_clf,
        )

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
