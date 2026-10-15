#!/usr/bin/env python3
"""
luk_scanner.py - momentum scanner modelled on Martin Luk's published process.

What it scans (end-of-day data from Massive with --massive-key, otherwise Yahoo Finance via yfinance):
  1. LEADERS        up >= 30% over 1, 3 or 6 months, ADR >= 5%, liquid, above the 50 EMA,
                    ranked by relative strength (top 2% flagged).
  2. POTENT         today's strongest movers on above-average volume (shows rotation/themes).
  3. EP             gap-ups of >= 5% on >= 2.5x volume that held (episodic pivots).
  4. PULLBACK LONG  a leader in an uptrend that flushed below support (9/21/50 EMA or an AVWAP)
                    and reclaimed it, closing in the upper half of the day's range.
                    Also a WATCH list of leaders pulling into support but not yet flushed.
  5. BREAKOUT       a leader coiling: 9/21/50 EMAs converged, inside day or tight range,
                    near highs. Trigger is a break of today's high.
  6. PULLBACK SHORT a stock that closed below a declining EMA and bounced back into it
                    (9 EMA "first bounce", or 21/50 EMA / AVWAP-from-high in a downtrend),
                    then got rejected. Trigger is a break of today's low.
  + MARKET REGIME   QQQ / IWM / SPY vs their EMAs, and leader-list expansion vs your last run.
  + SIZING          0.5% account risk per trade, 35% max position, scaled down for very high ADR
                    and for weaker regimes.

Usage:
  pip install yfinance pandas numpy
  python luk_scanner.py                          # built-in momentum universe (~120 names, fast)
  python luk_scanner.py --universe all           # every US-listed common stock (slow, ~10-15 min)
  python luk_scanner.py --universe tickers.txt   # your own list, one ticker per line
  python luk_scanner.py --tickers MU,SNDK,WDC    # quick check of a few names
  python luk_scanner.py --account 50000 --risk 0.5
  python luk_scanner.py --premarket              # add a pre-market gap scan (run 8:00-9:29 ET)
  python luk_scanner.py --export-data            # also save luk_data.csv for the web scanner

Whole market from Massive (formerly Polygon.io), free plan works:
  python luk_scanner.py --massive-key YOUR_KEY                 # every US stock and ADR
  python luk_scanner.py --massive-key YOUR_KEY --export-data   # same, plus luk_data.csv for the website
  (or set MASSIVE_API_KEY once instead of passing --massive-key)
  The first run downloads ~300 days of whole-market data: about an hour on the free plan
  (5 calls/minute). Later runs fetch only the new day. Data is cached in ./luk_cache/massive/.
  On a paid plan add --massive-plan paid to skip the wait between calls.

Run it after the close for clean daily bars; intraday runs use a partial last bar.
Results print to the console and are saved as CSVs plus a TradingView watchlist in
./luk_scans/scan_YYYY-MM-DD/.

The daily scan finds candidates; Luk times entries on intraday charts (opening-range high,
1/5-minute reclaim, prior hourly low for shorts). Review every chart before trading.
This is an educational tool, not financial advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_UNIVERSE = """
NVDA AMD AVGO MU SNDK WDC STX TSM ARM SMCI ANET MRVL QCOM INTC AMAT LRCX KLAC ASML ON MPWR
CRDO ALAB COHR LITE CIEN VRT ETN GEV CEG VST TLN OKLO SMR NNE LEU CCJ BWXT
PLTR TSLA META MSFT GOOGL AMZN AAPL NFLX ORCL CRWD PANW NET DDOG SNOW MDB ZS NOW SHOP APP RDDT
HOOD COIN MSTR MARA RIOT CLSK IREN CORZ CIFR WULF HUT GLXY CRCL SOFI AFRM UPST LMND NU TOST
IONQ RGTI QBTS QUBT RKLB ASTS LUNR PL RDW ACHR JOBY SOUN BBAI AI PATH TEM HIMS CVNA UBER DASH
DKNG RBLX GME AMC TTD IOT DUOL ONON CELH ELF LLY NVO VKTX AXON NBIS CRWV
""".split()

INDEXES = ["QQQ", "IWM", "SPY"]


@dataclass
class Config:
    min_price: float = 5.0
    min_dollar_vol: float = 20e6      # 20-day average dollar volume
    min_adr: float = 5.0              # Luk: ADR above 5%
    momentum_pct: float = 30.0        # up >= 30% over 1, 3 or 6 months
    rs_top_pct: float = 2.0           # flag the top 2% by relative strength
    flush_tol: float = 0.003          # low must reach support * (1 + 0.3%)
    tight_long_risk: float = 4.0      # pullback entries with stop <= 4% are flagged "tight"
    touch_tol: float = 0.01           # short: high within 1% of resistance counts as a touch
    short_min_adr: float = 3.5        # shorts: skip names that barely move (e.g. buyout targets)
    short_max_risk: float = 8.0       # shorts: skip if the stop (today's high) is more than 8% away
    ema_converge_pct: float = 4.0     # max spread between 9/21/50 EMA for a coil
    ep_gap_pct: float = 5.0
    ep_rvol: float = 2.5
    potent_min_chg: float = 4.0
    potent_top_n: int = 30
    account: float = 100_000.0
    risk_pct: float = 0.5             # % of account risked per trade
    max_position_pct: float = 35.0    # max position size as % of account


# --------------------------------------------------------------------------- helpers

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def pct_ret(c: pd.Series, n: int) -> float:
    if len(c) <= n or c.iloc[-1 - n] <= 0:
        return np.nan
    return (c.iloc[-1] / c.iloc[-1 - n] - 1) * 100


def avwap(df: pd.DataFrame, pos: int) -> pd.Series:
    """Anchored VWAP from bar `pos` (integer position) to the last bar."""
    seg = df.iloc[pos:]
    tp = (seg["High"] + seg["Low"] + seg["Close"]) / 3
    vol = seg["Volume"].replace(0, np.nan).fillna(0)
    cv = vol.cumsum().replace(0, np.nan)
    return (tp * vol).cumsum() / cv


def position_size(cfg: Config, entry: float, stop: float, adr: float, mult: float) -> tuple[int, float]:
    """Shares and position % of account for a given entry/stop, Luk-style."""
    per_share = abs(entry - stop)
    if not (per_share > 0 and entry > 0):
        return 0, 0.0
    risk_dollars = cfg.account * cfg.risk_pct / 100 * mult
    shares = math.floor(risk_dollars / per_share)
    cap_pct = cfg.max_position_pct * (min(1.0, 8.0 / adr) if adr and adr > 8 else 1.0)
    cap_shares = math.floor(cfg.account * cap_pct / 100 / entry)
    shares = max(0, min(shares, cap_shares))
    return shares, round(shares * entry / cfg.account * 100, 1)


# --------------------------------------------------------------------------- per-ticker analysis

def analyze(ticker: str, df: pd.DataFrame, cfg: Config) -> dict | None:
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < 130:
        return None
    df["Volume"] = df["Volume"].fillna(0)
    o, h, l, c, v = (df[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    e9, e21, e50 = ema(c, 9), ema(c, 21), ema(c, 50)

    close, high, low, opn = c.iloc[-1], h.iloc[-1], l.iloc[-1], o.iloc[-1]
    prev_close = c.iloc[-2]
    E9, E21, E50 = e9.iloc[-1], e21.iloc[-1], e50.iloc[-1]
    slope9 = E9 - e9.iloc[-2]
    slope21 = E21 - e21.iloc[-6]
    slope50 = E50 - e50.iloc[-6]

    adr = ((h / l - 1) * 100).rolling(20).mean().iloc[-1]
    dollar_vol = (c * v).rolling(20).mean().iloc[-1]
    vol50 = v.iloc[:-1].rolling(50).mean().iloc[-1]
    rvol = v.iloc[-1] / vol50 if vol50 and vol50 > 0 else np.nan

    r21, r63, r126 = pct_ret(c, 21), pct_ret(c, 63), pct_ret(c, 126)
    r252 = pct_ret(c, 252)
    rs_raw = 0.4 * r63 + 0.3 * r126 + 0.3 * (r252 if not np.isnan(r252) else r126)
    chg = (close / prev_close - 1) * 100
    gap = (opn / prev_close - 1) * 100
    day_range = high - low
    cpos = (close - low) / day_range if day_range > 0 else 0.5

    look = min(252, len(df))
    hi_pos = len(df) - look + int(np.argmax(h.iloc[-look:].to_numpy()))
    lo_look = min(126, len(df))
    lo_pos = len(df) - lo_look + int(np.argmin(l.iloc[-lo_look:].to_numpy()))
    hi_52 = h.iloc[hi_pos]
    off_high = (close / hi_52 - 1) * 100
    days_since_high = len(df) - 1 - hi_pos

    # AVWAPs: from the 52-week high, from the 6-month low, from the latest EP (gap + volume) day
    av_hi = avwap(df, hi_pos).iloc[-1] if days_since_high >= 3 else np.nan
    av_hi_prev = avwap(df, hi_pos).iloc[-6] if days_since_high >= 8 else np.nan
    av_lo = avwap(df, lo_pos).iloc[-1] if len(df) - 1 - lo_pos >= 3 else np.nan
    gaps = (o / c.shift(1) - 1) * 100
    vavg = v.shift(1).rolling(50).mean()
    ep_days = np.where((gaps >= cfg.ep_gap_pct) & (v >= 2 * vavg))[0]
    ep_days = [p for p in ep_days if len(df) - 64 <= p <= len(df) - 4]
    av_ep = avwap(df, ep_days[-1]).iloc[-1] if ep_days else np.nan

    inside = high <= h.iloc[-2] and low >= l.iloc[-2]
    range_pct = day_range / low * 100 if low > 0 else np.nan
    tight_day = inside or (range_pct < 0.7 * adr)
    ema_spread = (max(E9, E21, E50) - min(E9, E21, E50)) / close * 100

    liquid = close >= cfg.min_price and dollar_vol >= cfg.min_dollar_vol
    momentum = any(x >= cfg.momentum_pct for x in (r21, r63, r126) if not np.isnan(x))
    leader = liquid and momentum and adr >= cfg.min_adr and close > E50

    # former leader: doubled at some point in the last 6 months vs the 6 months before
    former_leader = False
    if len(c) >= 252:
        prior_low = c.iloc[-252:-126].min()
        former_leader = prior_low > 0 and c.iloc[-126:].max() / prior_low >= 2.0

    row = dict(
        ticker=ticker, close=round(close, 2), chg_pct=round(chg, 2), gap_pct=round(gap, 2),
        rvol=round(rvol, 2) if not np.isnan(rvol) else np.nan, adr_pct=round(adr, 2),
        dollar_vol_m=round(dollar_vol / 1e6, 1), ret_1m=round(r21, 1), ret_3m=round(r63, 1),
        ret_6m=round(r126, 1) if not np.isnan(r126) else np.nan, rs_raw=rs_raw,
        off_52w_high_pct=round(off_high, 1), days_since_high=days_since_high,
        ema9=round(E9, 2), ema21=round(E21, 2), ema50=round(E50, 2),
        avwap_high=round(av_hi, 2) if not np.isnan(av_hi) else np.nan,
        avwap_low=round(av_lo, 2) if not np.isnan(av_lo) else np.nan,
        avwap_ep=round(av_ep, 2) if not np.isnan(av_ep) else np.nan,
        ema_spread_pct=round(ema_spread, 2), inside_day=inside,
        liquid=liquid, leader=leader, former_leader=former_leader, setups=[],
        hi_pos=int(hi_pos), lo_pos=int(lo_pos), last_date=df.index[-1],
    )

    supports = {"9EMA": E9, "21EMA": E21, "50EMA": E50,
                "AVWAP_high": av_hi, "AVWAP_low": av_lo, "AVWAP_EP": av_ep}
    supports = {k: x for k, x in supports.items() if x is not None and not np.isnan(x)}

    # ---- pullback long: flush below support, reclaim, strong close
    # 21 EMA above a rising 50 EMA (the 21 often flattens during a pullback, so its slope isn't required)
    uptrend = close > E50 and E21 > E50 and slope50 > 0
    if leader and uptrend:
        reclaimed = [k for k, lvl in supports.items() if low <= lvl * (1 + cfg.flush_tol) and close > lvl]
        if reclaimed and cpos >= 0.5:
            key_lvl = min(supports[k] for k in reclaimed)
            confluence = sum(1 for x in supports.values() if abs(x / key_lvl - 1) <= 0.015)
            risk = (close - low) / close * 100
            row["setups"].append(dict(
                setup="PULLBACK_LONG", entry=close, stop=low, risk_pct=risk,
                note=f"reclaimed {'+'.join(reclaimed)}; confluence {confluence}"
                     + ("; TIGHT" if risk <= cfg.tight_long_risk else ""),
                confluence=confluence))
        else:
            near = [k for k, lvl in supports.items() if 0 <= (low / lvl - 1) <= 0.03]
            if near and close < h.iloc[-5:].max():
                row["setups"].append(dict(
                    setup="PULLBACK_WATCH", entry=np.nan, stop=np.nan, risk_pct=np.nan,
                    note=f"pulling into {'+'.join(near)}; wait for an intraday flush and reclaim",
                    confluence=len(near)))

    # ---- breakout coil: converged EMAs, tight day, near highs
    if (leader and close >= E21 * 0.98 and ema_spread <= cfg.ema_converge_pct
            and tight_day and off_high >= -15):
        risk = (high - low) / high * 100
        row["setups"].append(dict(
            setup="BREAKOUT", entry=high, stop=low, risk_pct=risk,
            note=("inside day" if inside else "tight range") + f"; EMA spread {ema_spread:.1f}%; "
                 "trigger above today's high (or the 5-min opening-range high)",
            confluence=np.nan))

    # ---- episodic pivot
    if liquid and gap >= cfg.ep_gap_pct and rvol >= cfg.ep_rvol and close >= opn:
        row["setups"].append(dict(
            setup="EP", entry=close, stop=low, risk_pct=(close - low) / close * 100,
            note=f"gap {gap:.1f}% on {rvol:.1f}x volume, held the open", confluence=np.nan))

    # ---- pullback short: bounce into declining resistance, rejected
    # Only names that actually move (no buyout targets pinned at a deal price) and that are
    # either a former leader breaking down or already in a downtrend.
    downtrend = close < E50 and slope21 < 0 and slope50 < 0
    if liquid and adr >= cfg.short_min_adr and (former_leader or downtrend):
        res = []
        # "first bounce": the close crossed below the 9 EMA within the last 5 days
        cv, ev = c.to_numpy(), e9.to_numpy()
        fresh_break = any(cv[i - 1] >= ev[i - 1] and cv[i] < ev[i] for i in range(len(cv) - 6, len(cv) - 1))
        if slope9 < 0 and fresh_break and high >= E9 * (1 - cfg.touch_tol) and close < E9:
            res.append("9EMA")
        if downtrend:
            if high >= E21 * (1 - cfg.touch_tol) and close < E21:
                res.append("21EMA")
            if high >= E50 * (1 - cfg.touch_tol) and close < E50:
                res.append("50EMA")
        if (not np.isnan(av_hi) and not np.isnan(av_hi_prev) and av_hi < av_hi_prev
                and close < E21 and high >= av_hi * (1 - cfg.touch_tol) and close < av_hi):
            res.append("AVWAP_high")
        risk = (high - low) / low * 100
        if res and cpos <= 0.5 and risk <= cfg.short_max_risk:
            row["setups"].append(dict(
                setup="PULLBACK_SHORT", entry=low, stop=high, risk_pct=risk,
                note=f"rejected at declining {'+'.join(res)}; trigger below today's low "
                     "(intraday: prior hourly low or failed VWAP retest)"
                     + ("; FORMER LEADER" if former_leader else ""),
                confluence=len(res)))
    return row


# --------------------------------------------------------------------------- market regime

def market_regime(index_data: dict[str, pd.DataFrame]) -> tuple[str, float, pd.DataFrame]:
    rows, score = [], 0
    for sym in INDEXES:
        df = index_data.get(sym)
        if df is None or len(df) < 60:
            continue
        c = df["Close"].dropna()
        e9, e21, e50 = ema(c, 9), ema(c, 21), ema(c, 50)
        last = c.iloc[-1]
        a9, a21, a50 = last > e9.iloc[-1], last > e21.iloc[-1], last > e50.iloc[-1]
        rising21 = e21.iloc[-1] > e21.iloc[-6]
        if sym in ("QQQ", "IWM"):
            score += 1 if (a21 and rising21) else 0
            score -= 1 if not a50 else 0
        rows.append(dict(index=sym, close=round(last, 2), above_9ema=a9, above_21ema=a21,
                         ema21_rising=rising21, above_50ema=a50))
    if score >= 2:
        label, mult = "GREEN - full risk on longs", 1.0
    elif score >= 0:
        label, mult = "YELLOW - test the waters (half risk), be selective", 0.5
    else:
        label, mult = "RED - quarter risk on longs; pullback shorts favoured", 0.25
    return label, mult, pd.DataFrame(rows)


# --------------------------------------------------------------------------- scan

def run_scan(data: dict[str, pd.DataFrame], cfg: Config,
             index_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame | str | float]:
    rows = []
    for t, df in data.items():
        try:
            r = analyze(t, df, cfg)
        except Exception as exc:  # one bad ticker shouldn't stop the scan
            print(f"  skipped {t}: {exc}", file=sys.stderr)
            continue
        if r:
            rows.append(r)
    if not rows:
        return {"empty": True}
    base = pd.DataFrame(rows)
    base["rs_pctile"] = base["rs_raw"].rank(pct=True).mul(100).round(1)
    base["rs_top"] = base["rs_pctile"] >= 100 - cfg.rs_top_pct

    label, mult, regime_df = market_regime(index_data)

    leader_cols = ["ticker", "close", "rs_pctile", "rs_top", "ret_1m", "ret_3m", "ret_6m", "adr_pct",
                   "dollar_vol_m", "off_52w_high_pct", "ema9", "ema21", "ema50", "avwap_high", "avwap_low"]
    leaders = base[base["leader"]].sort_values("rs_pctile", ascending=False)[leader_cols]

    potent = base[base["liquid"] & (base["chg_pct"] >= cfg.potent_min_chg) & (base["rvol"] >= 1.5)]
    potent = potent.sort_values("chg_pct", ascending=False).head(cfg.potent_top_n)[
        ["ticker", "close", "chg_pct", "rvol", "adr_pct", "rs_pctile", "leader"]]

    setup_rows = []
    for _, r in base.iterrows():
        for s in r["setups"]:
            short = s["setup"] == "PULLBACK_SHORT"
            m = 1.0 if short else mult
            shares, pos_pct = (0, 0.0)
            if not np.isnan(s["entry"]):
                shares, pos_pct = position_size(cfg, s["entry"], s["stop"], r["adr_pct"], m)
            setup_rows.append(dict(
                setup=s["setup"], ticker=r["ticker"], close=r["close"],
                entry=round(s["entry"], 2) if not np.isnan(s["entry"]) else np.nan,
                stop=round(s["stop"], 2) if not np.isnan(s["stop"]) else np.nan,
                risk_pct=round(s["risk_pct"], 2) if not np.isnan(s["risk_pct"]) else np.nan,
                shares=shares, position_pct=pos_pct, adr_pct=r["adr_pct"], rs_pctile=r["rs_pctile"],
                confluence=s["confluence"], note=s["note"]))
    setups = pd.DataFrame(setup_rows)

    def pick(name, sort, asc):
        if setups.empty:
            return pd.DataFrame()
        return setups[setups["setup"] == name].sort_values(sort, ascending=asc).reset_index(drop=True)

    return dict(
        regime=label, regime_mult=mult, regime_table=regime_df, all=base, leaders=leaders,
        potent=potent,
        ep=pick("EP", "rs_pctile", False),
        pullback_long=pick("PULLBACK_LONG", "risk_pct", True),
        pullback_watch=pick("PULLBACK_WATCH", "rs_pctile", False),
        breakout=pick("BREAKOUT", "risk_pct", True),
        pullback_short=pick("PULLBACK_SHORT", "risk_pct", True),
    )


# --------------------------------------------------------------------------- data

def load_all_us() -> list[str]:
    urls = ["https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"]
    out = []
    for url in urls:
        df = pd.read_csv(url, sep="|")
        df = df[~df.iloc[:, 0].astype(str).str.startswith("File Creation Time")]
        sym_col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
        df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")]
        bad = df["Security Name"].astype(str).str.contains(
            r"Warrant|Right|Unit|Preferred|Depositary Share|Notes|Debenture", case=False, regex=True)
        syms = df.loc[~bad, sym_col].astype(str)
        syms = syms[~syms.str.contains(r"[\$\^]", regex=True)].str.replace(".", "-", regex=False)
        out += syms.tolist()
    return sorted(set(out))


def _extract(data: pd.DataFrame, t: str) -> pd.DataFrame | None:
    if isinstance(data.columns, pd.MultiIndex):
        if t in data.columns.get_level_values(0):
            df = data[t]
        elif t in data.columns.get_level_values(1):
            df = data.xs(t, axis=1, level=1)
        else:
            return None
    else:
        df = data
    df = df.dropna(how="all")
    return df if len(df) else None


def download(tickers: list[str], period: str = "2y", chunk: int = 150) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        print(f"  downloading {i + 1}-{i + len(part)} of {len(tickers)}...")
        try:
            data = yf.download(part, period=period, interval="1d", auto_adjust=True,
                               group_by="ticker", threads=True, progress=False)
        except Exception as exc:
            print(f"  chunk failed: {exc}", file=sys.stderr)
            continue
        for t in part:
            df = _extract(data, t)
            if df is not None:
                out[t] = df
        if len(tickers) > chunk:
            time.sleep(1)
    return out


def premarket_scan(tickers: list[str], daily: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Gap-ups in pre-market (run 8:00-9:29 ET). Needs yfinance intraday data with prepost=True."""
    import yfinance as yf
    rows = []
    for i in range(0, len(tickers), 100):
        part = tickers[i:i + 100]
        try:
            data = yf.download(part, period="5d", interval="5m", prepost=True,
                               group_by="ticker", threads=True, progress=False)
        except Exception as exc:
            print(f"  pre-market chunk failed: {exc}", file=sys.stderr)
            continue
        for t in part:
            df = _extract(data, t)
            if df is None or t not in daily:
                continue
            idx = df.index.tz_convert("America/New_York") if df.index.tz else df.index
            df = df.set_axis(idx)
            today = idx[-1].date()
            pm = df[(df.index.date == today) & (df.index.time < dt.time(9, 30))]
            if pm.empty:
                continue
            d = daily[t]
            prev = d[d.index.date < today]["Close"].dropna()
            if prev.empty:
                continue
            last, pc = pm["Close"].dropna().iloc[-1], prev.iloc[-1]
            gap = (last / pc - 1) * 100
            pm_dollar = float((pm["Close"] * pm["Volume"]).sum())
            if gap >= 4 and pm_dollar >= 1e6:
                rows.append(dict(ticker=t, prev_close=round(pc, 2), premarket=round(last, 2),
                                 gap_pct=round(gap, 2), pm_dollar_vol_m=round(pm_dollar / 1e6, 2)))
        time.sleep(1)
    return pd.DataFrame(rows).sort_values("gap_pct", ascending=False) if rows else pd.DataFrame()


# --------------------------------------------------------------------------- output

# --------------------------------------------------------------------------- Massive (formerly Polygon.io)

class MassiveError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class Massive:
    """Whole-market daily data from Massive, cached locally so each run only fetches new days.

    Free (Basic) plan: 5 calls/minute, end-of-day data, 2 years of history. The first run
    downloads about `days` trading days (~1 call each, ~12 s apart); later runs fetch only
    the days that are missing. Prices are stored unadjusted and split-adjusted at load time,
    so a new split never leaves stale history in the cache.
    """
    BASE = "https://api.massive.com"
    KEEP_TYPES = {"CS", "ADRC"}   # common stocks and ADRs; drops ETFs, warrants, preferreds, units

    def __init__(self, key: str, plan: str = "basic", cache_dir: Path = Path("luk_cache") / "massive",
                 fetch=None, sleep=time.sleep, today: dt.date | None = None):
        self.key = key
        self.min_interval = 12.5 if plan == "basic" else 0.0
        self.cache = cache_dir
        (self.cache / "daily").mkdir(parents=True, exist_ok=True)
        self._fetch = fetch or self._http
        self._sleep = sleep
        self._last = 0.0
        self.calls = 0
        if today is None:
            # Latest US session that could have closed: before 5 pm New York time, start from yesterday.
            ny = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=5)
            today = ny.date() if ny.hour >= 17 else ny.date() - dt.timedelta(days=1)
        self.today = today

    # ---- HTTP
    def _http(self, url: str) -> tuple[int, dict]:
        import json
        import urllib.error
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "luk-scanner/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode("utf-8"))
            except Exception:
                body = {"message": str(e)}
            return e.code, body
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            return 599, {"message": f"network problem: {e}"}   # e.g. the computer slept or Wi-Fi dropped

    def get(self, path_or_url: str, params: dict | None = None) -> dict:
        from urllib.parse import urlencode
        url = path_or_url if path_or_url.startswith("http") else self.BASE + path_or_url
        q = dict(params or {})
        if "apiKey=" not in url:
            q["apiKey"] = self.key
        if q:
            url += ("&" if "?" in url else "?") + urlencode(q)
        for attempt in range(6):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            status, body = self._fetch(url)
            if status == 429:                       # rate limited: back off and retry
                self._sleep(60 if attempt < 5 else 0)
                continue
            if status == 599:                       # network dropped: wait and retry
                print(f"  Massive: {body['message']}; retrying in 30 s...")
                self._sleep(30 if attempt < 5 else 0)
                continue
            if status >= 400:
                msg = body.get("message") or body.get("error") or body.get("status") or "request failed"
                raise MassiveError(status, str(msg))
            return body
        raise MassiveError(status, f"gave up after several retries ({body.get('message') or body.get('error') or 'rate limited'})")

    # ---- reference data
    def ticker_types(self) -> dict[str, str]:
        import json
        f = self.cache / "tickers.json"
        if f.exists():
            saved = json.loads(f.read_text())
            if (self.today - dt.date.fromisoformat(saved["fetched"])).days < 7:
                return saved["types"]
        print("  Massive: refreshing the list of stock tickers (about 12 calls, weekly)...")
        types, url, params = {}, "/v3/reference/tickers", {"market": "stocks", "active": "true", "limit": 1000, "sort": "ticker", "order": "asc"}
        while url:
            body = self.get(url, params)
            for r in body.get("results", []):
                types[r["ticker"]] = r.get("type", "")
            url, params = body.get("next_url"), None
        f.write_text(json.dumps({"fetched": self.today.isoformat(), "types": types}))
        return types

    def splits(self, since: dt.date) -> pd.DataFrame:
        rows, url = [], "/stocks/v1/splits"
        params = {"execution_date.gte": since.isoformat(), "limit": 5000, "sort": "execution_date.asc"}
        try:
            body = self.get(url, params)
        except MassiveError as e:
            if e.status != 400:
                raise
            body = self.get(url, {"limit": 5000, "sort": "execution_date.desc"})   # filter locally instead
        while True:
            rows += body.get("results", [])
            nxt = body.get("next_url")
            if not nxt:
                break
            body = self.get(nxt)
        df = pd.DataFrame(rows, columns=["ticker", "execution_date", "split_from", "split_to"])
        if df.empty:
            return df
        df = df.dropna()
        df["execution_date"] = pd.to_datetime(df["execution_date"])
        return df[df["execution_date"] >= pd.Timestamp(since)]

    # ---- daily bars
    def _day_files(self, d: dt.date) -> tuple[Path, Path]:
        return self.cache / "daily" / f"{d.isoformat()}.csv.gz", self.cache / "daily" / f"{d.isoformat()}.empty"

    def update(self, days: int = 300) -> None:
        """Make sure the cache holds the last `days` trading days, fetching what's missing."""
        weekdays, d = [], self.today
        while len(weekdays) < int(days * 1.04) + 6:      # ~4% of weekdays are market holidays
            if d.weekday() < 5:
                weekdays.append(d)
            d -= dt.timedelta(days=1)
        missing = [d for d in weekdays if not any(p.exists() for p in self._day_files(d))]
        if not missing:
            return
        mins = len(missing) * self.min_interval / 60
        print(f"  Massive: fetching {len(missing)} day(s) of whole-market data"
              + (f" (about {mins:.0f} min on the free plan; later runs only fetch new days)" if mins >= 1 else "") + "...")
        for i, d in enumerate(missing, 1):            # newest first
            data_f, empty_f = self._day_files(d)
            try:
                body = self.get(f"/v2/aggs/grouped/locale/us/market/stocks/{d.isoformat()}",
                                {"adjusted": "false", "include_otc": "false"})
            except MassiveError as e:
                if e.status in (401,):
                    raise SystemExit("Massive rejected the API key. Check it at massive.com and try again.")
                if e.status in (403,):
                    if (self.today - d).days <= 5:
                        print(f"  Massive: {d} isn't available on your plan yet; skipping it for now.")
                        continue
                    print(f"  Massive: {d} is outside your plan's history ({e}); stopping the backfill there.")
                    break
                print(f"  Massive: {d} failed ({e}); will retry next run.")
                continue
            res = body.get("results") or []
            if res:
                df = pd.DataFrame(res)[["T", "o", "h", "l", "c", "v"]]
                df.to_csv(data_f, index=False, compression="gzip")
            elif (self.today - d).days > 5:            # old weekday with no data = market holiday
                empty_f.touch()
            # recent empty days are left unmarked: end-of-day data may not be published yet
            if i % 20 == 0 or i == len(missing):
                print(f"    {i}/{len(missing)} days done")

    def load(self, universe: set[str] | None, days: int = 300) -> dict[str, pd.DataFrame]:
        """Return {ticker: OHLCV DataFrame}, split-adjusted. universe=None means every stock + ADR."""
        self.update(days)
        files = sorted((self.cache / "daily").glob("*.csv.gz"))[-days:]
        if not files:
            return {}
        if universe is None:
            types = self.ticker_types()
            keep = {t for t, ty in types.items() if ty in self.KEEP_TYPES} | set(INDEXES)
        else:
            keep = set(universe) | set(INDEXES)
        frames = []
        for f in files:
            df = pd.read_csv(f, dtype={"T": str})
            df = df[df["T"].isin(keep)]
            df["date"] = pd.Timestamp(f.name[:10])
            frames.append(df)
        allrows = pd.concat(frames, ignore_index=True).dropna(subset=["o", "h", "l", "c"])
        allrows = allrows.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        out = {t: g.set_index("date").sort_index()[["Open", "High", "Low", "Close", "Volume"]]
               for t, g in allrows.groupby("T", sort=False)}

        first, last = allrows["date"].min(), allrows["date"].max()
        sp = self.splits(first.date())
        applied = 0
        for _, s in sp.iterrows():
            t, ex = s["ticker"], s["execution_date"]
            if t not in out or ex > last or not (s["split_from"] > 0 and s["split_to"] > 0):
                continue
            f = float(s["split_from"]) / float(s["split_to"])     # 2-for-1: from 1, to 2 -> prices x 0.5
            df = out[t]
            before = df.index < ex
            if before.any():
                df.loc[before, ["Open", "High", "Low", "Close"]] *= f
                df.loc[before, "Volume"] /= f
                applied += 1
        print(f"  Massive: {len(out)} tickers, {len(files)} trading days through {last.date()}"
              f"{f', {applied} split adjustment(s)' if applied else ''} ({self.calls} API calls this run).")
        return out


def show(title: str, df: pd.DataFrame, cols: list[str] | None = None, n: int = 25) -> None:
    print(f"\n=== {title} ({0 if df is None else len(df)}) ===")
    if df is None or df.empty:
        print("  none")
        return
    print((df[cols] if cols else df).head(n).to_string(index=False))


def leader_breadth(root: Path, today: str, count: int) -> tuple[str, int | None, str | None]:
    """Record today's leader count; return (summary text, previous count, previous date)."""
    hist = root / "leader_history.csv"
    prev, prev_date = None, None
    root.mkdir(parents=True, exist_ok=True)
    if hist.exists():
        h = pd.read_csv(hist, dtype={"date": str})
        h = h[h["date"] != today]
        earlier = h[h["date"] < today]
        if len(earlier):
            prev, prev_date = int(earlier.iloc[-1]["leaders"]), str(earlier.iloc[-1]["date"])
    else:
        h = pd.DataFrame(columns=["date", "leaders"])
    h = pd.concat([h, pd.DataFrame([{"date": today, "leaders": count}])], ignore_index=True)
    h.sort_values("date").to_csv(hist, index=False)
    if prev is None:
        return f"{count} leaders (first run; expansion/contraction shows from the next run)", None, None
    trend = "EXPANDING" if count > prev else "CONTRACTING" if count < prev else "flat"
    return f"{count} leaders vs {prev} on {prev_date} -> {trend}", prev, prev_date


# --------------------------------------------------------------------------- published results

def _num(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def _ms(ts) -> int:
    ts = pd.Timestamp(ts)
    return int(pd.Timestamp(ts.date()).tz_localize("UTC").timestamp() * 1000)


def write_site_results(res: dict, data: dict[str, pd.DataFrame], index_data: dict[str, pd.DataFrame],
                       cfg: Config, path: Path, breadth: tuple[int, int | None, str | None],
                       always: list[str] | None = None) -> int:
    """results.json for the web page: every list, the market read, and price bars for each listed stock."""
    import json
    base = res["all"].set_index("ticker", drop=False)
    names = {"PULLBACK_LONG": "pullbackLong", "BREAKOUT": "breakout", "EP": "ep",
             "PULLBACK_SHORT": "pullbackShort", "PULLBACK_WATCH": "watch"}

    def row_json(t):
        r = base.loc[t]
        return dict(
            ticker=t, date=_ms(r["last_date"]), close=_num(r["close"]), chg=_num(r["chg_pct"]), gap=_num(r["gap_pct"]),
            rvol=_num(r["rvol"]), adr=_num(r["adr_pct"]), dollarVol=_num(r["dollar_vol_m"] * 1e6, 0),
            r21=_num(r["ret_1m"]), r63=_num(r["ret_3m"]), r126=_num(r["ret_6m"]), rsPctile=_num(r["rs_pctile"], 1),
            rsTop=bool(r["rs_top"]), offHigh=_num(r["off_52w_high_pct"]), daysSinceHigh=int(r["days_since_high"]),
            E9=_num(r["ema9"]), E21=_num(r["ema21"]), E50=_num(r["ema50"]), avHi=_num(r["avwap_high"]),
            avLo=_num(r["avwap_low"]), avEp=_num(r["avwap_ep"]), emaSpread=_num(r["ema_spread_pct"]),
            inside=bool(r["inside_day"]), liquid=bool(r["liquid"]), leader=bool(r["leader"]),
            formerLeader=bool(r["former_leader"]), hiPos=int(r["hi_pos"]), loPos=int(r["lo_pos"]),
            setups=[dict(setup=s["setup"], entry=_num(s["entry"]), stop=_num(s["stop"]), riskPct=_num(s["risk_pct"]),
                         confluence=_num(s["confluence"]), note=s["note"]) for s in r["setups"]])

    lists, tickers = {}, set()
    for key, name in [("pullback_long", "PULLBACK_LONG"), ("breakout", "BREAKOUT"), ("ep", "EP"),
                      ("pullback_short", "PULLBACK_SHORT"), ("pullback_watch", "PULLBACK_WATCH")]:
        df = res[key]
        items = []
        if isinstance(df, pd.DataFrame) and not df.empty:
            for s in df.itertuples():
                items.append(dict(setup=name, ticker=s.ticker, entry=_num(s.entry), stop=_num(s.stop),
                                  riskPct=_num(s.risk_pct), confluence=_num(s.confluence), note=s.note))
                tickers.add(s.ticker)
        lists[names[name]] = items
    lists["leaders"] = res["leaders"]["ticker"].tolist()
    lists["potent"] = res["potent"]["ticker"].tolist() if len(res["potent"]) else []
    tickers |= set(lists["leaders"]) | set(lists["potent"]) | {t for t in (always or []) if t in base.index}

    def bars(df):
        d = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        return dict(t=[_ms(x) for x in d.index], o=[_num(x) for x in d["Open"]], h=[_num(x) for x in d["High"]],
                    l=[_num(x) for x in d["Low"]], c=[_num(x) for x in d["Close"]], v=[_num(x, 0) for x in d["Volume"]])

    label, mult, tab = res["regime"], res["regime_mult"], res["regime_table"]
    level = "green" if label.startswith("GREEN") else "yellow" if label.startswith("YELLOW") else "red"
    now, prev, prev_date = breadth
    out = dict(
        version=1, generated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        lastDate=_ms(base["last_date"].max()), scanned=int(len(base)),
        cfg=dict(account=cfg.account, riskPct=cfg.risk_pct, maxPositionPct=cfg.max_position_pct, minAdr=cfg.min_adr,
                 momentumPct=cfg.momentum_pct, minDollarVol=cfg.min_dollar_vol, minPrice=cfg.min_price,
                 flushTol=cfg.flush_tol, touchTol=cfg.touch_tol, emaConvergePct=cfg.ema_converge_pct,
                 epGapPct=cfg.ep_gap_pct, epRvol=cfg.ep_rvol, shortMinAdr=cfg.short_min_adr,
                 shortMaxRisk=cfg.short_max_risk),
        regime=dict(level=level, label=level.capitalize(), mult=mult, known=bool(len(tab)),
                    rows=[dict(sym=r["index"], close=_num(r["close"]), a9=bool(r["above_9ema"]), a21=bool(r["above_21ema"]),
                               rising21=bool(r["ema21_rising"]), a50=bool(r["above_50ema"])) for _, r in tab.iterrows()]),
        breadth=dict(now=now, prev=prev, prevDate=prev_date),
        lists=lists,
        rows={t: row_json(t) for t in sorted(tickers) if t in base.index},
        bars={**{t: bars(data[t]) for t in sorted(tickers) if t in data},
              **{t: bars(d) for t, d in index_data.items()}},
    )
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(json.dumps(out, separators=(",", ":")))
    return len(out["rows"])


def write_job_summary(res: dict, scan_date: str, breadth_text: str) -> None:
    """Short Markdown summary for the GitHub Actions run page (only when running on GitHub)."""
    import os
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return
    lines = [f"## Luk scan for {scan_date}", "", f"**Market:** {res['regime']}", "", f"**Leaders:** {breadth_text}", ""]
    for key, title in [("pullback_long", "Pullback long"), ("breakout", "Breakout"), ("ep", "EP"),
                       ("pullback_short", "Pullback short")]:
        df = res[key]
        lines.append(f"### {title} ({0 if df is None or df.empty else len(df)})")
        if df is None or df.empty:
            lines += ["None today.", ""]
            continue
        lines += ["| Ticker | Close | Trigger | Stop | Risk | RS |", "|---|---:|---:|---:|---:|---:|"]
        for s in df.head(15).itertuples():
            lines.append(f"| {s.ticker} | {s.close:.2f} | {s.entry:.2f} | {s.stop:.2f} | {s.risk_pct:.1f}% | {s.rs_pctile:.0f} |")
        lines.append("")
    w = res["pullback_watch"]
    if isinstance(w, pd.DataFrame) and not w.empty:
        lines += [f"**Watch:** {', '.join(w['ticker'].head(20))}", ""]
    with open(target, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_prices(data: dict[str, pd.DataFrame], path: Path) -> None:
    """One long CSV (ticker,date,open,high,low,close,volume) the web scanner can load."""
    frames = []
    for t, df in data.items():
        d = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
        d.columns = ["open", "high", "low", "close", "volume"]
        d.insert(0, "date", pd.to_datetime(d.index).strftime("%Y-%m-%d"))
        d.insert(0, "ticker", t)
        frames.append(d)
    if frames:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames).round(4).to_csv(path, index=False)
        print(f"Saved prices for {len(frames)} tickers to {path.resolve()} (load this file in the web scanner)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Martin Luk-style momentum scanner")
    ap.add_argument("--universe", default=None,
                    help="'default', 'all', or a path to a ticker file (default: 'all' with Massive, else the built-in list)")
    ap.add_argument("--massive-key", default=None,
                    help="Massive API key (or set the MASSIVE_API_KEY environment variable) to scan the whole market")
    ap.add_argument("--massive-plan", choices=["basic", "paid"], default="basic",
                    help="'basic' waits 12.5 s between calls (free plan limit); 'paid' doesn't wait")
    ap.add_argument("--massive-days", type=int, default=300, help="trading days of history to keep (default 300)")
    ap.add_argument("--tickers", help="comma-separated tickers (overrides --universe)")
    ap.add_argument("--account", type=float, default=100_000)
    ap.add_argument("--risk", type=float, default=0.5, help="%% of account risked per trade")
    ap.add_argument("--min-adr", type=float, default=5.0)
    ap.add_argument("--min-dollar-vol", type=float, default=20.0, help="millions, 20-day average")
    ap.add_argument("--premarket", action="store_true")
    ap.add_argument("--export-data", action="store_true",
                    help="also save all downloaded prices to luk_data.csv for the web scanner")
    ap.add_argument("--out", default="luk_scans")
    ap.add_argument("--chart-tickers", default="",
                    help="comma-separated stocks to always include on the web page, e.g. MU,SNDK,NVDA")
    ap.add_argument("--site", default=None,
                    help="also write results.json into this folder for the web page (used by the GitHub setup)")
    a = ap.parse_args()

    cfg = Config(account=a.account, risk_pct=a.risk, min_adr=a.min_adr, min_dollar_vol=a.min_dollar_vol * 1e6)

    import os
    key = a.massive_key or os.environ.get("MASSIVE_API_KEY")
    universe = a.universe or ("all" if key else "default")

    tickers: list[str] | None
    if a.tickers:
        tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    elif universe == "default":
        tickers = DEFAULT_UNIVERSE
    elif universe == "all":
        tickers = None
    else:
        tickers = [x.strip().upper() for x in Path(universe).read_text().split() if x.strip()]

    if key:
        print("Loading prices from Massive...")
        mv = Massive(key, plan=a.massive_plan)
        data = mv.load(None if tickers is None else set(tickers), days=a.massive_days)
        print(f"Scanning {len(set(data) - set(INDEXES))} stocks...")
    else:
        if tickers is None:
            print("Loading the full US stock list from nasdaqtrader.com...")
            tickers = load_all_us()
        tickers = sorted(set(tickers) - set(INDEXES))
        print(f"Scanning {len(tickers)} tickers...")
        data = download(tickers + INDEXES)
    if a.export_data:
        export_prices(data, Path(a.out) / "luk_data.csv")
    index_data = {k: data.pop(k) for k in INDEXES if k in data}
    res = run_scan(data, cfg, index_data)
    if res.get("empty"):
        print("No usable data came back. Check your connection or the ticker list.")
        return

    today = pd.Timestamp(res["all"]["last_date"].max()).date().isoformat()   # date of the latest bar
    root = Path(a.out)
    out = root / f"scan_{today}"
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== MARKET REGIME ===")
    print(res["regime_table"].to_string(index=False))
    print(f"Regime: {res['regime']}")
    breadth_text, prev, prev_date = leader_breadth(root, today, len(res["leaders"]))
    print(f"Leader breadth: {breadth_text}")
    print("Also check your own equity curve: if your last trades keep failing, cut risk regardless.")

    setup_cols = ["ticker", "close", "entry", "stop", "risk_pct", "shares", "position_pct", "adr_pct",
                  "rs_pctile", "note"]
    show("LEADERS (top by relative strength)", res["leaders"],
         ["ticker", "close", "rs_pctile", "rs_top", "ret_1m", "ret_3m", "ret_6m", "adr_pct", "off_52w_high_pct"])
    show("POTENT - today's strongest movers", res["potent"])
    show("EP - gap-ups on volume", res["ep"], setup_cols)
    show("PULLBACK LONG - flush and reclaim", res["pullback_long"], setup_cols)
    show("PULLBACK WATCH - approaching support", res["pullback_watch"], ["ticker", "close", "adr_pct", "rs_pctile", "note"])
    show("BREAKOUT - coiled near highs", res["breakout"], setup_cols)
    show("PULLBACK SHORT - rejected at declining resistance", res["pullback_short"], setup_cols)

    for name in ["leaders", "potent", "ep", "pullback_long", "pullback_watch", "breakout", "pullback_short"]:
        df = res[name]
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(out / f"{name}.csv", index=False)
    res["all"].drop(columns=["setups"]).to_csv(out / "all_metrics.csv", index=False)

    if a.site:
        always = [t.strip().upper() for t in a.chart_tickers.split(",") if t.strip()]
        n = write_site_results(res, data, index_data, cfg, Path(a.site), (len(res["leaders"]), prev, prev_date), always)
        print(f"Wrote results for {n} stocks to {Path(a.site).resolve() / 'results.json'}")
    write_job_summary(res, today, breadth_text)

    if a.premarket:
        pm = premarket_scan(list(data.keys()), data, cfg)
        show("PRE-MARKET GAPS", pm)
        if not pm.empty:
            pm.to_csv(out / "premarket.csv", index=False)

    # TradingView import format: one ###section line per list
    sections = []
    for name, label in [("pullback_long", "PULLBACK_LONG"), ("breakout", "BREAKOUT"), ("ep", "EP"),
                        ("pullback_watch", "WATCH"), ("pullback_short", "SHORT"), ("leaders", "LEADERS")]:
        df = res[name]
        if isinstance(df, pd.DataFrame) and not df.empty:
            sections.append(f"###{label}," + ",".join(df["ticker"].drop_duplicates().tolist()))
    (out / "tradingview_watchlist.txt").write_text(",".join(sections))
    print(f"\nSaved CSVs and a TradingView watchlist to {out.resolve()}")


if __name__ == "__main__":
    main()
