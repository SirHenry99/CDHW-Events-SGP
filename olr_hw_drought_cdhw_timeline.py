# ─────────────────────────────────────────────────────────────────────────────
# ENSO (daily SST → Niño-3.4) × Daily SPI from PRISM precip × HW × Drought × CDHW
# Drought rule: SPI ≤ DOY-P20 for ≥10 consecutive days
# HW rule: Tmax ≥ DOY-P90 for ≥5 consecutive days
# CSVs include avg_SPI (drought), avg_Tmax (HW), avg_SPI & avg_Tmax (CDHW)
# Region: SGP (25–39N, −105 to −90E). Period: 1981-01-01 … 2022-12-31
# ─────────────────────────────────────────────────────────────────────────────
import os, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import BrokenBarHCollection
import matplotlib.dates as mdates
from scipy.stats import gamma as sgamma, norm

# ============================================================================
# 0) CONFIG
# ============================================================================
# ► INPUTS
SST_DIR     = r"D:/sst"               # daily/subdaily SST files (var SST_VARN)
SST_VARN    = "sst"

PREC_DIR    = r"D:/realprism/"        # PRISM precip files (daily or subdaily)
PREC_VARN   = "prec"

TMAX_DIR    = r"D:/prismmaxtemp1/"    # PRISM Tmax (Kelvin → will convert to °C)
TMAX_VARN   = "tmax"

# ► Spatial subset (SGP)
LAT_MIN, LAT_MAX = 25.0, 39.0
LON_MIN, LON_MAX = -105.0, -90.0

# ► Time window
TIME_START = "1981-01-01"
TIME_END   = "2022-12-31"
idx_full   = pd.date_range(TIME_START, TIME_END, freq="D")

# ► DOY thresholds (smoothed)
SMOOTH_DOY_WINDOW = 7   # ±7 days for threshold smoothing
TMAX_PCTL = 90.0        # HW day flag: Tmax ≥ P90_DOY
SPI_PCTL  = 20.0        # Drought day flag: SPI ≤ P20_DOY

# ► SPI settings
SPI_WINDOW_DAYS = 30    # rolling accumulation window for daily SPI
DOY_SMOOTH_FIT  = 15    # ±days around DOY to enlarge sample for gamma fit

# ► Event minimum lengths
MINLEN_DROUGHT = 10     # days
MINLEN_HW      = 5      # days

# ► ENSO phase thresholds (applied to daily Niño-3.4 anomalies)
ELNINO_THRESH  =  0.5
LANINA_THRESH  = -0.5
# Niño-3.4 box (0–360 lon)
N34_LAT_MIN, N34_LAT_MAX = -5.0, 5.0
N34_LON_MIN, N34_LON_MAX = 190.0, 240.0

# ► Output
OUT_DIR  = "./enso_hw_drought_outputs"
OUT_PLOT = os.path.join(OUT_DIR, "enso_hw_drought_cdhw_timeline.png")
os.makedirs(OUT_DIR, exist_ok=True)

# ► Performance / file matcher
DTYPE = np.float32
YR_PAT = re.compile(r"(198[1-9]|199[0-9]|20[0-1][0-9]|202[0-2])")

print("✅ Configuration loaded.")


# ============================================================================
# 1) HELPERS
# ============================================================================
def open_and_concat(varname, directory, chunks=None):
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory)
                    if f.endswith((".nc", ".nc4")) and YR_PAT.search(f)])
    if not files:
        raise FileNotFoundError(f"No NetCDF files 1981–2022 found in {directory}")
    ds = xr.open_mfdataset(files, combine="by_coords", chunks=chunks or {}, parallel=False)
    if varname not in ds:
        raise KeyError(f"Variable '{varname}' not found. Available: {list(ds.data_vars)}")
    return ds

def standardize_lat_lon(ds):
    # unify coord names to lat/lon
    rename = {}
    if "latitude" in ds.coords: rename["latitude"] = "lat"
    if "longitude" in ds.coords: rename["longitude"] = "lon"
    if "Lat" in ds.coords: rename["Lat"] = "lat"
    if "Lon" in ds.coords: rename["Lon"] = "lon"
    if "lat" not in ds.coords or "lon" not in ds.coords:
        ds = ds.rename(rename)
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("Dataset must contain 'lat' and 'lon'.")
    return ds

def sel_sgp(da):
    # handles ascending/descending latitude
    lat = da["lat"].values
    lat_slice = slice(LAT_MAX, LAT_MIN) if lat[0] > lat[-1] else slice(LAT_MIN, LAT_MAX)
    return da.sel(lon=slice(LON_MIN, LON_MAX), lat=lat_slice)

def area_mean_1d(da):
    return da.mean([d for d in da.dims if d != "time"])

def doy_percentile(series: pd.Series, q: float, smooth_window=7):
    s = series.copy()
    # Map Feb 29 → Feb 28
    leap = (s.index.month == 2) & (s.index.day == 29)
    if leap.any():
        s.loc[leap] = s.loc[(s.index.month == 2) & (s.index.day == 28)].mean()

    doy = s.index.dayofyear
    doy = np.where(doy == 366, 365, doy)
    tmp = pd.DataFrame({"v": s.values, "doy": doy})
    pct = tmp.groupby("doy")["v"].quantile(q/100.0).astype(DTYPE)

    if smooth_window and smooth_window > 0:
        k = smooth_window
        pct_ext = pd.concat([pct.iloc[-k:], pct, pct.iloc[:k]])
        pct = pct_ext.rolling(2*k+1, center=True, min_periods=1).mean().iloc[k:-k].astype(DTYPE)

    return pd.Series(pct.reindex(tmp["doy"]).values, index=s.index, dtype=DTYPE)

def spans_from_flags_with_minlen(flags_series: pd.Series, min_len: int):
    """Return list of (start, end) contiguous-True spans with min length."""
    spans, in_span, start = [], False, None
    for i, (ts, val) in enumerate(flags_series.items()):
        if val and not in_span:
            in_span, start = True, ts
        elif (not val or i == len(flags_series)-1) and in_span:
            end = flags_series.index[i-1] if not val else flags_series.index[i]
            duration = (end - start).days + 1
            if duration >= min_len:
                spans.append((start, end))
            in_span = False
    return spans

def spans_to_df(spans, name, df_ref, add_cols=None):
    """Build event dataframe with start/end/duration + optional averages from df_ref columns."""
    out = pd.DataFrame(spans, columns=["event_start", "event_end"])
    out["duration_days"] = (out["event_end"] - out["event_start"]).dt.days + 1
    out["event"] = name
    if add_cols:
        for label, colname in add_cols.items():
            vals = []
            for s, e in spans:
                seg = df_ref.loc[s:e, colname]
                vals.append(np.float32(np.nanmean(seg.values)))
            out[label] = vals
    return out


# ============================================================================
# 2) ENSO DAILY INDEX FROM DAILY SST (Niño-3.4 DOY anomalies)
# ============================================================================
print("\n📥 Loading SST and computing daily Niño-3.4 anomalies → ENSO index…")
sst_ds = open_and_concat(SST_VARN, SST_DIR, chunks={"time": 50})
sst_ds = standardize_lat_lon(sst_ds)
# make longitudes 0–360 if needed
if sst_ds.lon.max() <= 180:
    sst_ds = sst_ds.assign_coords(lon=(sst_ds.lon % 360)).sortby("lon")

sst = sst_ds[SST_VARN].astype(DTYPE).sel(time=slice(TIME_START, TIME_END))
sst1d = sst.resample(time="1D").mean()
sst1d = sst1d.sel(time=~((sst1d.time.dt.month == 2) & (sst1d.time.dt.day == 29)))

lat_vals = sst1d["lat"].values
lat_slice = slice(N34_LAT_MAX, N34_LAT_MIN) if lat_vals[0] > lat_vals[-1] else slice(N34_LAT_MIN, N34_LAT_MAX)
sst34 = sst1d.sel(lat=lat_slice, lon=slice(N34_LON_MIN, N34_LON_MAX)).mean(["lat", "lon"]).astype(DTYPE)

climo34 = sst34.groupby("time.dayofyear").mean("time").astype(DTYPE)
doy = xr.where(sst34.time.dt.dayofyear == 366, 365, sst34.time.dt.dayofyear)
sst34_anom = (sst34.groupby(doy) - climo34)
enso_daily = sst34_anom.rename("ENSO").to_series()
enso_daily.index = pd.to_datetime(enso_daily.index)
enso_daily = enso_daily.reindex(idx_full).interpolate(limit_direction="both").astype(DTYPE)

sst_ds.close(); del sst_ds, sst, sst1d, sst34, sst34_anom, climo34
print(f"   ENSO daily ready: {enso_daily.index[0].date()} → {enso_daily.index[-1].date()} ({len(enso_daily)} days)")

# ============================================================================
# 3) LOAD PRISM PRECIP (→ daily area mean) AND TMAX (→ °C daily area mean)
# ============================================================================
def load_daily_series(dir_path, varname, transform=None, desc=""):
    files = sorted([f for f in os.listdir(dir_path) if f.endswith((".nc", ".nc4")) and YR_PAT.search(f)])
    print(f"   Found {len(files)} {desc} files.")
    pieces = []
    for fn in tqdm(files, desc=f"Reading {desc} files"):
        ds = xr.open_dataset(os.path.join(dir_path, fn))
        ds = standardize_lat_lon(ds)
        da = ds[varname].astype(DTYPE)
        da = sel_sgp(da)
        if transform is not None:
            da = transform(da)
        da = da.resample(time="1D").mean()
        da = area_mean_1d(da)
        s = pd.Series(da.values, index=pd.to_datetime(da.time.values), name=varname)
        pieces.append(s)
        ds.close()
    ser = pd.concat(pieces).sort_index()
    ser = ser.loc[TIME_START:TIME_END]
    ser = ser.reindex(idx_full).interpolate(limit_direction="both").astype(DTYPE)
    return ser

print("\n🌧️ Loading PRISM precip …")
prec = load_daily_series(PREC_DIR, PREC_VARN, transform=None, desc="precip").rename("prec")

print("\n🌡️ Loading PRISM Tmax (K→°C) …")
tmax_c = load_daily_series(
    TMAX_DIR, TMAX_VARN,
    transform=lambda da: da - 273.15,
    desc="Tmax"
).rename("tmax_c")

# ─────────────────────────────────────────────────────────────────────────────
def compute_daily_spi(prec_series: pd.Series, window: int = 30, doy_smooth: int = 15) -> pd.Series:
    # 30-day rolling accumulation
    acc = prec_series.rolling(window=window, min_periods=window).sum()

    # Remove Feb 29 by mapping to Feb 28 for grouping convenience
    idx = acc.index
    is_feb29 = (idx.month == 2) & (idx.day == 29)
    if is_feb29.any():
        acc.loc[is_feb29] = acc.loc[(idx.month == 2) & (idx.day == 28)].mean()

    # Precompute DOY (365 only)
    doy = idx.dayofyear
    doy = np.where(doy == 366, 365, doy)

    spi_vals = np.full(acc.size, np.nan, dtype=np.float32)

    # Build lookup from DOY -> indices
    doy_to_indices = {d: np.where(doy == d)[0] for d in range(1, 366)}

    # Helper to get training sample indices for a DOY ±k (with wrap)
    def training_indices_for_doy(d, k):
        window_days = [(d + off - 1) % 365 + 1 for off in range(-k, k + 1)]
        idxs = np.concatenate([doy_to_indices[w] for w in window_days])
        # We train only on non-NaN accumulations
        idxs = idxs[~np.isnan(acc.values[idxs])]
        return idxs

    # Iterate through DOYs and fit once per DOY (vectorize assignment)
    for d in tqdm(range(1, 366), desc="Fitting gamma per DOY (SPI)"):
        train_idx = training_indices_for_doy(d, doy_smooth)
        if train_idx.size < 30:  # not enough samples; fall back to empirical z-score
            # simple rank-based z for this DOY as a fallback
            for ii in doy_to_indices[d]:
                x = acc.values[ii]
                if np.isnan(x): 
                    continue
                ref = acc.values[train_idx]
                r = (np.sum(ref <= x) + 0.5) / (ref.size + 1.0)
                spi_vals[ii] = np.float32(norm.ppf(r))
            continue

        ref = acc.values[train_idx].astype(np.float64)
        # Mixed distribution: probability of zero
        p0 = np.mean(ref == 0.0)
        pos = ref[ref > 0.0]

        if pos.size < 5:
            # almost all zeros → assign very low SPI when x==0, else rank-based
            for ii in doy_to_indices[d]:
                x = acc.values[ii]
                if np.isnan(x): 
                    continue
                if x == 0:
                    spi_vals[ii] = np.float32(norm.ppf(p0/2.0 + 1e-6))
                else:
                    r = (np.sum(ref <= x) + 0.5) / (ref.size + 1.0)
                    spi_vals[ii] = np.float32(norm.ppf(r))
            continue

        # Fit gamma (MLE) to positive values; use floc=0 to avoid negative support
        try:
            k_hat, loc_hat, theta_hat = sgamma.fit(pos, floc=0.0)
            # For each target day with this DOY, compute SPI
            for ii in doy_to_indices[d]:
                x = acc.values[ii]
                if np.isnan(x):
                    continue
                if x <= 0.0:
                    cdf = p0  # all mass at/below zero goes to the zero probability
                else:
                    Gx = sgamma.cdf(x, k_hat, loc=0.0, scale=theta_hat)
                    cdf = p0 + (1.0 - p0) * Gx
                # clamp CDF in (0,1) to avoid infs
                cdf = min(max(cdf, 1e-7), 1.0 - 1e-7)
                spi_vals[ii] = np.float32(norm.ppf(cdf))
        except Exception:
            # Fallback: empirical rank → z
            for ii in doy_to_indices[d]:
                x = acc.values[ii]
                if np.isnan(x): 
                    continue
                r = (np.sum(ref <= x) + 0.5) / (ref.size + 1.0)
                spi_vals[ii] = np.float32(norm.ppf(r))

    spi = pd.Series(spi_vals, index=idx, name=f"SPI{window}")
    return spi

print("\n📐 Computing daily SPI from precip…")
spi = compute_daily_spi(prec, window=SPI_WINDOW_DAYS, doy_smooth=DOY_SMOOTH_FIT)


# ============================================================================
print("\n🧮 Computing DOY thresholds & day flags…")
tmax_thr = doy_percentile(tmax_c, TMAX_PCTL, SMOOTH_DOY_WINDOW)
spi_thr  = doy_percentile(spi,    SPI_PCTL,  SMOOTH_DOY_WINDOW)

is_HW_day      = (tmax_c >= tmax_thr).astype(np.uint8)
is_Drought_day = (spi    <= spi_thr ).astype(np.uint8)

# Unified daily frame (no CDHW here yet)
df = pd.DataFrame({
    "ENSO":        enso_daily.values,
    "Tmax_C":      tmax_c.values,
    "SPI":         spi.values,
    "HW_day":      is_HW_day.values,
    "Drought_day": is_Drought_day.values,
}, index=idx_full)

# ============================================================================
MINLEN_DROUGHT = 10
MINLEN_HW      = 5
MINLEN_CDHW    = 2  # your requirement

print("\n📦 Building events with minimum-length rules…")

# 6.1 Build *qualified* drought & heatwave events from daily flags
drought_spans = spans_from_flags_with_minlen(df["Drought_day"].astype(bool), MINLEN_DROUGHT)
hw_spans      = spans_from_flags_with_minlen(df["HW_day"].astype(bool),       MINLEN_HW)

def spans_to_mask(index: pd.DatetimeIndex, spans):
    """Boolean mask of days that lie within any provided spans."""
    mask = pd.Series(False, index=index)
    for s, e in spans:
        mask.loc[s:e] = True
    return mask

# Masks marking days that are inside qualified events
drought_event_mask = spans_to_mask(df.index, drought_spans)
hw_event_mask      = spans_to_mask(df.index, hw_spans)

# 6.2 CDHW qualified days = intersection of the qualified-event masks
is_CDHW_qualified_day = (drought_event_mask & hw_event_mask).astype(np.uint8)

# Extract CDHW event spans with minimum 2-day length
cdhw_spans = spans_from_flags_with_minlen(is_CDHW_qualified_day.astype(bool), MINLEN_CDHW)

# 6.3 Convert spans to event tables with requested averages
drought_events = spans_to_df(
    drought_spans, "Drought", df,
    add_cols={"avg_SPI": "SPI"}
)
hw_events = spans_to_df(
    hw_spans, "HW", df,
    add_cols={"avg_Tmax": "Tmax_C"}
)
cdhw_events = spans_to_df(
    cdhw_spans, "CDHW", df,
    add_cols={"avg_SPI": "SPI", "avg_Tmax": "Tmax_C"}
)

# 6.4 Update daily DF with final (qualified) CDHW_day and save daily CSV
df["CDHW_day"] = is_CDHW_qualified_day.values

daily_path = os.path.join(OUT_DIR, "daily_timeseries_flags_with_spi.csv")
df.to_csv(daily_path)
print(f"   ✔ Saved daily time series & flags (with qualified CDHW) → {daily_path}")

# 6.5 Save event CSVs
drought_path = os.path.join(OUT_DIR, "events_Drought.csv")
hw_path      = os.path.join(OUT_DIR, "events_HW.csv")
cdhw_path    = os.path.join(OUT_DIR, "events_CDHW.csv")

drought_events.to_csv(drought_path, index=False)
hw_events.to_csv(hw_path, index=False)
cdhw_events.to_csv(cdhw_path, index=False)

print(f"   ✔ Saved events CSVs:")
print(f"     • Drought (≥{MINLEN_DROUGHT}d): {drought_path}  — {len(drought_events)} events")
print(f"     • HW (≥{MINLEN_HW}d):          {hw_path}       — {len(hw_events)} events")
print(f"     • CDHW (≥{MINLEN_CDHW}d):      {cdhw_path}     — {len(cdhw_events)} events")

# (Optional) quick peek
if len(drought_events):
    print("\n   Drought head:\n", drought_events.head(3))
if len(hw_events):
    print("\n   HW head:\n", hw_events.head(3))
if len(cdhw_events):
    print("\n   CDHW head:\n", cdhw_events.head(3))

# ============================================================================
# 7) PUBLICATION PLOT — ENSO (top, radiant blue) + bold HW/Drought/CDHW rails
# ============================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Patch
from matplotlib.transforms import blended_transform_factory
from matplotlib.gridspec import GridSpec

print("\n🎨 Creating publication-quality figure with custom colors…")

# ------------------ Style & Layout ------------------
mpl.rcParams.update({
    "figure.dpi": 700,
    "savefig.dpi": 850,
    "figure.figsize": (25, 10),
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 17,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.6,
})

fig = plt.figure(constrained_layout=False)
gs  = GridSpec(nrows=2, ncols=1, height_ratios=[2.2, 1.0], hspace=0.08, figure=fig)

ax_top = fig.add_subplot(gs[0, 0])   # ENSO
ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)  # Rails

# ------------------ Top Panel: ENSO ------------------
enso_vals = df["ENSO"].values.astype(np.float32)
# Extended y-axis limits to ±3.0
ymin, ymax = -3.0, 3.2

ax_top.set_xlim(df.index[0], df.index[-1])
ax_top.set_ylim(ymin, ymax)

# Bright radiant sea blue ENSO line
ax_top.plot(
    df.index, enso_vals,
    lw=1.8, color="#0025ff", alpha=0.9,
    label="ENSO (Niño-3.4 daily anomaly)"
)

# Reference lines with legend labels
ref_line_0 = ax_top.axhline(0.0, ls="--", lw=1.5, color="k", alpha=0.35, label="Neutral Threshold")
ref_line_elnino = ax_top.axhline(ELNINO_THRESH, ls=":", lw=1.5, color="#FF6347", alpha=0.7, label="El Niño Threshold")  # tomato red
ref_line_lanina = ax_top.axhline(LANINA_THRESH, ls=":", lw=1.5, color="#4682B4", alpha=0.7, label="La Niña Threshold")  # steel blue

# Grid for readability
ax_top.grid(axis="y", ls=":", alpha=0.5)

# Labels & title - MOVED TITLE TO TOP
ax_top.set_ylabel("ENSO (Niño-3.4 anomaly, °C)", fontweight='bold')
ax_top.set_title("(a) ENSO", fontsize=20, 
                loc='left', fontweight='bold', pad=10)  # Added title for top panel

# Panel tag
#ax_top.text(0.01, 0.96, "(a) ENSO", transform=ax_top.transAxes,
#            ha="left", va="top", fontsize=13, weight="bold")

# ------------------ Bottom Panel: Event Rails ------------------
ax_bot.set_ylim(0, 1)
ax_bot.set_yticks([])
ax_bot.set_ylabel("")

# Clean x-axis formatting
ax_bot.xaxis.set_major_locator(mdates.YearLocator(2))
ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for label in ax_top.get_xticklabels():
    label.set_visible(False)

ax_bot.grid(axis="x", ls=":", alpha=0.35)

# Lane definitions with your requested colors
lanes = {
    "HW":      {"y0": 0.10, "h": 0.16, "color": "#D7191C", "label": "HW Events",      "spans": hw_spans},      # deep bright red
    "Drought": {"y0": 0.40, "h": 0.16, "color": "#8B4513", "label": "Drought Events", "spans": drought_spans}, # deep bright brown
    "CDHW":    {"y0": 0.70, "h": 0.16, "color": "#000000", "label": "CDHW Events",     "spans": cdhw_spans},    # black
}

# ADDED TITLE FOR BOTTOM PANEL
ax_bot.set_title("(b) Events", fontsize=20,  
                loc='left',fontweight='bold', pad=1.5)

# blended transform (x in data coords, y in axes coords)
x_data__y_axes = blended_transform_factory(ax_bot.transData, ax_bot.transAxes)

legend_handles = []
for key, lane in lanes.items():
    y0_ax = lane["y0"]
    h_ax  = lane["h"]
    color = lane["color"]
    label = lane["label"]
    spans = lane["spans"]

    # faint lane background for clarity
    ax_bot.add_patch(Rectangle(
        (mdates.date2num(df.index[0]), y0_ax),
        width=mdates.date2num(df.index[-1]) - mdates.date2num(df.index[0]),
        height=h_ax, transform=x_data__y_axes,
        facecolor=color, alpha=0.08, edgecolor="none", zorder=1)
    )

    # bold event bars
    for s, e in spans:
        x0 = mdates.date2num(pd.to_datetime(s))
        x1 = mdates.date2num(pd.to_datetime(e) + pd.Timedelta(days=1))
        ax_bot.add_patch(Rectangle(
            (x0, y0_ax),
            width=x1 - x0,
            height=h_ax,
            transform=x_data__y_axes,
            facecolor=color,
            edgecolor="none",
            alpha=1.0,
            zorder=10,
            clip_on=False
        ))

    legend_handles.append(Patch(facecolor=color, edgecolor="none", alpha=1.0, label=label))
    
# Create OLR line handle for legend
enso_legend_handle = Line2D([0], [0], color='#0025ff', lw=1.8, alpha=0.9, label='ENSO')
# Panel tag
#ax_bot.text(0.01, 0.95, "(b) Events", transform=ax_bot.transAxes,
#            ha="left", va="top", fontsize=13, weight="bold")

# Create combined legend for both panels
all_legend_handles = [
    *legend_handles,  # Event rails
    ref_line_elnino,  # El Niño threshold
    ref_line_0,       # Neutral line
    ref_line_lanina,
    enso_legend_handle, # La Niña threshold
]

# Legend - MOVED OUTSIDE THE PLOT to upper center, above the top panel
leg = fig.legend(handles=all_legend_handles, ncol=7, 
                loc='lower center', bbox_to_anchor=(0.5, -0.02),
                frameon=True, fontsize=18)
for t in leg.get_texts():
    t.set_alpha(0.95)

# X-axis label
ax_bot.set_xlabel("YEARS", fontweight='bold')

# Tight layout & save high-quality versions
fig.tight_layout()
png_path = OUT_PLOT
svg_path = OUT_PLOT.replace(".png", ".svg")

fig.savefig(png_path, bbox_inches="tight", bbox_extra_artists=[leg])
fig.savefig(svg_path, bbox_inches="tight", bbox_extra_artists=[leg])

plt.close(fig)
print(f"🖼️ Figures saved → {png_path} and {svg_path}")


import netCDF4 as nc
import numpy as np

#file_path = "C:/Users/Henry O. Olayiwola/Downloads/olr_mjo_proxy_mjjas_1979_2024.nc"
file_path = "C:/Users/Henry O. Olayiwola/Desktop/olr_mjo_proxy_janDec_1981_2022_optimized.nc"
# Open the netCDF file
try:
    with nc.Dataset(file_path, 'r') as dataset:
        print("=" * 60)
        print("NETCDF FILE INFORMATION")
        print("=" * 60)
        
        # Print global attributes
        print("\nGLOBAL ATTRIBUTES:")
        print("-" * 30)
        if dataset.ncattrs():
            for attr in dataset.ncattrs():
                print(f"{attr}: {getattr(dataset, attr)}")
        else:
            print("No global attributes found")
        
        # Print dimensions
        print("\nDIMENSIONS:")
        print("-" * 30)
        for dim_name, dim in dataset.dimensions.items():
            print(f"{dim_name}: {len(dim)} (unlimited: {dim.isunlimited()})")
        
        # Print variables and their details
        print("\nVARIABLES:")
        print("-" * 30)
        for var_name, var in dataset.variables.items():
            print(f"\nVariable: {var_name}")
            print(f"  Shape: {var.shape}")
            print(f"  Data type: {var.dtype}")
            
            # Print variable attributes
            if var.ncattrs():
                print("  Attributes:")
                for attr in var.ncattrs():
                    print(f"    {attr}: {getattr(var, attr)}")
            
            # Get the variable data
            data = var[:]
            
            # Print statistics for numeric variables
            if data.dtype.kind in 'iufc':  # integer, unsigned, float, complex
                print(f"  Min value: {np.nanmin(data):.6f}")
                print(f"  Max value: {np.nanmax(data):.6f}")
                print(f"  Mean value: {np.nanmean(data):.6f}")
                
                # Print first 10 values (flattened if multi-dimensional)
                flat_data = data.flatten()
                print(f"  First 10 values: {flat_data[:10]}")
            else:
                print("  (Non-numeric data type)")
                
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"Error reading file: {e}")

import xarray as xr
import pandas as pd

# Path to your saved OLR index file
#OLR_INDEX_FILE = "C:/Users/Henry O. Olayiwola/Downloads/olr_mjo_proxy_mjjas_1979_2024.nc"  # adjust path if needed
OLR_INDEX_FILE = "C:/Users/Henry O. Olayiwola/Desktop/olr_mjo_proxy_janDec_1981_2022_optimized.nc"
print(f"📂 Loading OLR index from {OLR_INDEX_FILE} ...")

# Open the NetCDF file
olr_ds = xr.open_dataset(OLR_INDEX_FILE)

# Detect and extract the correct variable (handles different naming cases)
olr_var = None
for nm in ["OLR_MJO_Proxy", "olr_index", "index", "OLR_Index"]:
    if nm in olr_ds:
        olr_var = nm
        break
if olr_var is None:
    # fallback: pick first 1D variable with 'time' dimension
    for nm in olr_ds.data_vars:
        if "time" in olr_ds[nm].dims and olr_ds[nm].ndim == 1:
            olr_var = nm
            break
if olr_var is None:
    raise KeyError("No suitable OLR index variable found in file!")

# Assign to a consistent variable name
olr_index = olr_ds[olr_var].astype("float32")
olr_index.name = "OLR_MJO_Proxy"

# Optionally trim to your analysis period
olr_index = olr_index.sel(time=slice("1981-01-01", "2022-12-31"))

print("✅ OLR index loaded successfully:")
print(f"   Variable: {olr_var}")
print(f"   Shape: {olr_index.shape}")
print(f"   Time range: {str(olr_index.time.min().values)[:10]} → {str(olr_index.time.max().values)[:10]}")

# Convert time for alignment step
olr_times = pd.DatetimeIndex(olr_index.time.values)
# ============================================================================
# 7) PUBLICATION PLOT — OLR INDEX (top, radiant color) + bold HW/Drought/CDHW rails
# ============================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Patch
from matplotlib.transforms import blended_transform_factory
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

print("\n🎨 Creating publication-quality figure with custom colors…")

# ------------------ Style & Layout ------------------
mpl.rcParams.update({
    "figure.dpi": 700,
    "savefig.dpi": 850,
    "figure.figsize": (25, 10),
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 17,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.6,
})

fig = plt.figure(constrained_layout=False)
gs  = GridSpec(nrows=2, ncols=1, height_ratios=[2.2, 1.0], hspace=0.08, figure=fig)

ax_top = fig.add_subplot(gs[0, 0])   # OLR Index
ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)  # Rails

# ------------------ Top Panel: OLR Index ------------------
# Convert OLR index to pandas Series for plotting
olr_series = olr_index.to_series()

# Set appropriate y-axis limits for OLR data
olr_min, olr_max = olr_series.min(), olr_series.max()
ymin, ymax = olr_min - 0.1*(olr_max-olr_min), olr_max + 0.1*(olr_max-olr_min)

ax_top.set_xlim(olr_series.index[0], olr_series.index[-1])
ax_top.set_ylim(ymin, ymax)

# Dark green color for OLR line
olr_line = ax_top.plot(
    olr_series.index, olr_series.values,
    lw=1.8, color="#006400", alpha=0.9,  # Dark green color
    label="OLR Index"
)

# Reference line at zero with legend label
ref_line_0 = ax_top.axhline(0.0, ls="--", lw=1.5, color="k", alpha=0.35, label="Zero Reference")

# Grid for readability
ax_top.grid(axis="y", ls=":", alpha=0.5)

# Labels & title - MOVED TITLE TO TOP
ax_top.set_ylabel("Standardized OLR Index", fontweight='bold')
ax_top.set_title("(a) OLR Index", fontsize=20, 
                loc='left', fontweight='bold', pad=10)  # Added title for top panel

# ------------------ Bottom Panel: Event Rails ------------------
ax_bot.set_ylim(0, 1)
ax_bot.set_yticks([])
ax_bot.set_ylabel("")

# Clean x-axis formatting
ax_bot.xaxis.set_major_locator(mdates.YearLocator(2))
ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for label in ax_top.get_xticklabels():
    label.set_visible(False)

ax_bot.grid(axis="x", ls=":", alpha=0.35)

# Lane definitions with your requested colors
lanes = {
    "HW":      {"y0": 0.10, "h": 0.16, "color": "#D7191C", "label": "HW Events",      "spans": hw_spans},      # deep bright red
    "Drought": {"y0": 0.40, "h": 0.16, "color": "#8B4513", "label": "Drought Events", "spans": drought_spans}, # deep bright brown
    "CDHW":    {"y0": 0.70, "h": 0.16, "color": "#000000", "label": "CDHW Events",     "spans": cdhw_spans},    # black
}

# ADDED TITLE FOR BOTTOM PANEL
ax_bot.set_title("(b) Events", fontsize=20,  
                loc='left',fontweight='bold', pad=1.5)

# blended transform (x in data coords, y in axes coords)
x_data__y_axes = blended_transform_factory(ax_bot.transData, ax_bot.transAxes)

legend_handles = []
for key, lane in lanes.items():
    y0_ax = lane["y0"]
    h_ax  = lane["h"]
    color = lane["color"]
    label = lane["label"]
    spans = lane["spans"]

    # faint lane background for clarity
    ax_bot.add_patch(Rectangle(
        (mdates.date2num(olr_series.index[0]), y0_ax),
        width=mdates.date2num(olr_series.index[-1]) - mdates.date2num(olr_series.index[0]),
        height=h_ax, transform=x_data__y_axes,
        facecolor=color, alpha=0.08, edgecolor="none", zorder=1)
    )

    # bold event bars
    for s, e in spans:
        x0 = mdates.date2num(pd.to_datetime(s))
        x1 = mdates.date2num(pd.to_datetime(e) + pd.Timedelta(days=1))
        ax_bot.add_patch(Rectangle(
            (x0, y0_ax),
            width=x1 - x0,
            height=h_ax,
            transform=x_data__y_axes,
            facecolor=color,
            edgecolor="none",
            alpha=1.0,
            zorder=10,
            clip_on=False
        ))

    legend_handles.append(Patch(facecolor=color, edgecolor="none", alpha=1.0, label=label))

# Create OLR line handle for legend
olr_legend_handle = Line2D([0], [0], color='#006400', lw=1.8, alpha=0.9, label='OLR Index')

# Create combined legend for both panels
all_legend_handles = [
    olr_legend_handle,  # OLR Index line
    *legend_handles,    # Event rails
    ref_line_0,         # Zero reference line
]

# Legend - MOVED OUTSIDE THE PLOT to upper center, above the top panel
leg = fig.legend(handles=all_legend_handles, ncol=5, 
                loc='lower center', bbox_to_anchor=(0.5, -0.02),
                frameon=True, fontsize=18)
for t in leg.get_texts():
    t.set_alpha(0.95)

# X-axis label
ax_bot.set_xlabel("YEARS", fontweight='bold')

# Tight layout & save high-quality versions
fig.tight_layout()
png_path = OUT_PLOT
svg_path = OUT_PLOT.replace(".png", ".svg")

fig.savefig(png_path, bbox_inches="tight", bbox_extra_artists=[leg])
fig.savefig(svg_path, bbox_inches="tight", bbox_extra_artists=[leg])

plt.close(fig)
print(f"🖼️ Figures saved → {png_path} and {svg_path}")
