#!/usr/bin/env python
"""
Descriptive figures for the Florida Deepbiosphere data.

Uses only the downloaded inputs (GBIF CSV, GADM, EPA ecoregions, WorldClim BioClim,
NAIP 2021 tiles + index); it does not need Build_Data.py or the label summary.

Figures
  01 density      observations and species richness per hexagon
  02 split        1-degree latitude train/test bands (as in Build_Data.make_spatial_split)
  03 naip         NAIP 2021 tile footprints and which observations they cover
  04 climate      climate space: Florida land vs observations (BIO1 x BIO12, and PCA of all 19)
  05 ecoregions   sampling effort per EPA Level III ecoregion
  06 uncertainty  coordinate uncertainty of the records
  07 years        observations per year, iNaturalist vs other sources
  08 season       observations per month, and monthly profile of the top 20 species
  09 taxonomy     largest families and genera
  10 cutoff       species kept vs minimum-observation cutoff (200 and 500 marked)
  11 crops        example NAIP crops for a few species (needs the raw tiles)

Output: <out>/figures_fl.html (self-contained), <out>/png/*.png (300 dpi), <out>/tables/*.csv

Usage (inside the deepbio env):
  python figures_fl.py [--out DIR] [--only density,years,...] [--crop_species "Rhizophora mangle" ...]
"""
import argparse
import base64
import csv
import io
import json
import re
import time
import traceback
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.patches import Patch
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from rasterio.windows import Window

BASE = "/scratch/ajasapar/deepbiosphere"
EQ = "EPSG:3086"          # Florida GDL Albers, metres, equal area
VASCULAR = ['Gnetopsida', 'Liliopsida', 'Lycopodiopsida', 'Magnoliopsida',
            'Pinopsida', 'Polypodiopsida', 'Ginkgoopsida']
INAT = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"     # iNaturalist research-grade dataset key
KM_2_DEG = 1 / 111.32     # same approximation Build_Data uses for its 1.5 km split buffer
CROP_PX = 256             # Build_Data crop size in pixels
DEFAULT_CROPS = {"Rhizophora mangle": "red mangrove", "Serenoa repens": "saw palmetto",
                 "Pinus palustris": "longleaf pine", "Taxodium distichum": "bald cypress"}
BIO_NAMES = {1: "Annual mean temperature", 2: "Mean diurnal range", 3: "Isothermality",
             4: "Temperature seasonality", 5: "Max temp. of warmest month", 6: "Min temp. of coldest month",
             7: "Temperature annual range", 8: "Mean temp. of wettest quarter", 9: "Mean temp. of driest quarter",
             10: "Mean temp. of warmest quarter", 11: "Mean temp. of coldest quarter", 12: "Annual precipitation",
             13: "Precip. of wettest month", 14: "Precip. of driest month", 15: "Precipitation seasonality",
             16: "Precip. of wettest quarter", 17: "Precip. of driest quarter", 18: "Precip. of warmest quarter",
             19: "Precip. of coldest quarter"}
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

# styling: light surface, recessive grid, one sequential blue, blue<->red diverging
SURF, INK, INK2, GRID, LAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#5598e7", "#f0efec", "#ef8a89", "#a92a2a"])
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- loading

def load_occ(path, vascular=True):
    want = {"gbifID", "datasetKey", "class", "family", "genus", "species", "decimalLatitude",
            "decimalLongitude", "coordinateUncertaintyInMeters", "eventDate", "year", "month"}
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, usecols=lambda c: c in want,
                     dtype=str, on_bad_lines="skip")
    n_raw = len(df)
    for c in ["decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters", "year", "month"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "eventDate" in df:
        dt = pd.to_datetime(df["eventDate"].str[:10], errors="coerce")
        for c, v in [("year", dt.dt.year), ("month", dt.dt.month)]:
            df[c] = df[c].fillna(v) if c in df else v
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])
    df = df[df["species"].notna() & (df["species"].str.len() > 0)]
    n_sp = len(df)
    if vascular and "class" in df:
        df = df[df["class"].isin(VASCULAR)]
    df = df.reset_index(drop=True)
    x, y = Transformer.from_crs("EPSG:4326", EQ, always_xy=True).transform(
        df["decimalLongitude"].values, df["decimalLatitude"].values)
    df["x"], df["y"] = x, y
    df["sp_code"] = df["species"].astype("category").cat.codes
    return df, dict(raw=n_raw, with_species=n_sp, used=len(df))


def load_florida(base):
    hits = sorted(Path(base, "data/shpfiles/gadm36_USA").rglob("*_1.shp"))
    if hits:
        g = gpd.read_file(hits[0])
        g = g[g["NAME_1"] == "Florida"]
    else:   # fall back to the ecoregion outline
        g = gpd.read_file(Path(base, "data/shpfiles/ecoregions/fl/fl_eco_l3.shp"))
    g = g[["geometry"]].dissolve()
    return g.to_crs(EQ), g.to_crs("EPSG:4326")


def points(c):
    if "pts" not in c:
        o = c["occ"]
        c["pts"] = gpd.GeoDataFrame(geometry=gpd.points_from_xy(o["x"], o["y"]), crs=EQ)
    return c["pts"]


# ----------------------------------------------------------------------------- helpers

def map_axes(ax, fl):
    ax.set_aspect("equal")
    ax.set_axis_off()
    x0, y0, x1, y1 = fl.total_bounds
    bx, by = x0 + 0.04 * (x1 - x0), y0 + 0.06 * (y1 - y0)
    ax.plot([bx, bx + 1e5], [by, by], color=INK2, lw=2, solid_capstyle="butt")
    ax.text(bx + 5e4, by + 0.015 * (y1 - y0), "100 km", ha="center", va="bottom", fontsize=8, color=INK2)


def land(ax, fl):
    fl.plot(ax=ax, color=LAND, edgecolor="#c3c2b7", lw=0.5, zorder=0)


def hbar(ax, labels, values, color=BLUE, fmt="{:,.0f}"):
    yy = np.arange(len(values))[::-1]
    ax.barh(yy, values, color=color, height=0.72)
    ax.set_yticks(yy, labels)
    ax.grid(axis="y", visible=False)
    vmax = max(values) if len(values) else 1
    for yv, v in zip(yy, values):
        ax.text(v + vmax * 0.01, yv, fmt.format(v), va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, vmax * 1.15)


def save_table(c, name, df):
    df.to_csv(c["out"] / "tables" / f"{name}.csv", index=False)


def table_html(df, floatfmt="{:,.1f}"):
    def fmt(v):
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


# ----------------------------------------------------------------------------- figures
# Each returns (fig, caption_html).

def fig_density(c):
    occ, fl = c["occ"], c["fl"]
    x0, y0, x1, y1 = fl.total_bounds
    gs = max(10, int(round((x1 - x0) / (c["hex_km"] * 1000))))
    ext = (x0, x1, y0, y1)
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.6))
    res = {}
    for ax, kind in zip(axs, ["obs", "rich"]):
        land(ax, fl)
        if kind == "obs":
            hb = ax.hexbin(occ["x"], occ["y"], gridsize=gs, extent=ext, mincnt=1, cmap=SEQ,
                           norm=LogNorm(), linewidths=0)
            ax.set_title("Observations per hexagon")
            lab = "observations"
        else:
            hb = ax.hexbin(occ["x"], occ["y"], C=occ["sp_code"], reduce_C_function=lambda v: len(np.unique(v)),
                           gridsize=gs, extent=ext, cmap=SEQ, norm=LogNorm(vmin=1), linewidths=0)
            ax.set_title("Species per hexagon")
            lab = "species"
        res[kind] = np.asarray(hb.get_array())
        cb = fig.colorbar(hb, ax=ax, shrink=0.55, pad=0.0)
        cb.set_label(f"{lab} (log scale)")
        cb.outline.set_visible(False)
        map_axes(ax, fl)
    hex_w = (x1 - x0) / gs / 1000
    n = len(res["obs"])
    o = np.sort(res["obs"])[::-1]
    top = o[: max(1, int(0.01 * n))].sum() / o.sum() * 100
    cap = (f"Hexagons about {hex_w:.1f} km across; only hexagons with at least one record are drawn ({n:,}). "
           f"The busiest 1% of hexagons hold {top:.0f}% of all observations. Richness tracks effort closely, "
           f"so most of the richness pattern is sampling, not biology.")
    return fig, cap


def fig_split(c):
    occ, fl = c["occ"], c["fl"]
    lat = occ["decimalLatitude"].values
    ex = KM_2_DEG * 1.5
    rows = []
    band = np.full(len(lat), -1)
    buffer = np.zeros(len(lat), bool)
    for i, b in enumerate(range(24, 31)):
        inb = (lat >= b) & (lat < b + 1)
        test = (lat > b + ex) & (lat < b + 1 - ex)
        train = (lat < b) | (lat > b + 1)
        band[inb] = i
        buffer |= inb & ~test
        sp_test = set(occ["species"].values[test])
        sp_train = set(occ["species"].values[train])
        rows.append({"band": f"{b}–{b + 1}°N", "test obs": int(test.sum()), "in 1.5 km buffer": int((inb & ~test).sum()),
                     "train obs": int(train.sum()), "test species": len(sp_test),
                     "test species absent from train": len(sp_test - sp_train)})
    tab = pd.DataFrame(rows)
    save_table(c, "split_bands", tab)
    c["tables"]["split"] = tab

    fig, ax = plt.subplots(figsize=(10, 8))
    land(ax, fl)
    keep = band >= 0
    for parity, col in [(0, "#86b6ef"), (1, "#1c5cab")]:
        m = keep & ~buffer & (band % 2 == parity)
        ax.scatter(occ["x"].values[m], occ["y"].values[m], s=0.3, c=col, lw=0, rasterized=True)
    ax.scatter(occ["x"].values[buffer], occ["y"].values[buffer], s=0.6, c=ORANGE, lw=0, rasterized=True)
    tr = Transformer.from_crs("EPSG:4326", EQ, always_xy=True)
    lons = np.linspace(-88, -79.5, 50)
    x0, y0, x1, y1 = fl.total_bounds
    for i, b in enumerate(range(24, 32)):
        xs, ys = tr.transform(lons, np.full_like(lons, b))
        ax.plot(xs, ys, color=INK2, lw=0.6, ls=(0, (4, 3)))
        if b < 31:
            xm, ym = tr.transform(-79.6, b + 0.5)
            ax.text(min(xm, x1 + 2e4), ym, f"fold {i}  ·  {rows[i]['test obs']:,} test",
                    fontsize=8, color=INK2, va="center")
    ax.set_xlim(x0 - 1e4, x1 + 2.2e5)
    ax.legend(handles=[Patch(color="#86b6ef", label="band (alternate shading)"), Patch(color="#1c5cab", label="band"),
                       Patch(color=ORANGE, label="within 1.5 km of a band edge (dropped from test)")],
              loc="lower left", bbox_to_anchor=(0, 0.12), fontsize=8, frameon=True, facecolor=SURF, edgecolor="none", framealpha=1)
    ax.set_title("Spatial cross-validation bands (1° of latitude)")
    map_axes(ax, fl)
    worst = tab.loc[tab["test obs"].idxmin()]
    cap = (f"Each 1° latitude band is the test set of one fold and everything outside it trains that fold; records within "
           f"1.5 km of a band edge are left out of the test set. Smallest test band: {worst['band']} "
           f"({worst['test obs']:,} observations). Table below; a band with 0 test observations would crash Build_Data.")
    return fig, cap


def fig_naip(c):
    occ, fl = c["occ"], c["fl"]
    hits = sorted(Path(c["base"], "data/shpfiles/naip_tiffs/fl_shpfl_2021").rglob("*.shp"))
    if not hits:
        raise FileNotFoundError("no NAIP index shapefile in data/shpfiles/naip_tiffs/fl_shpfl_2021")
    tiles = gpd.read_file(hits[0]).to_crs(EQ)
    pts = points(c)
    j = gpd.sjoin(pts, tiles[["geometry"]], predicate="within", how="inner")
    covered = np.zeros(len(pts), bool)
    covered[np.unique(j.index.values)] = True
    per_tile = j.groupby("index_right").size()
    fig, ax = plt.subplots(figsize=(10, 8))
    land(ax, fl)
    tiles.boundary.plot(ax=ax, color="#5598e7", lw=0.25)
    ax.scatter(occ["x"].values[covered], occ["y"].values[covered], s=0.2, c=INK2, lw=0, rasterized=True, alpha=0.5)
    ax.scatter(occ["x"].values[~covered], occ["y"].values[~covered], s=3, c=ORANGE, lw=0, rasterized=True)
    ax.legend(handles=[Patch(facecolor="none", edgecolor="#5598e7", label=f"NAIP 2021 tile ({len(tiles):,})"),
                       Patch(color=INK2, label=f"observation inside a tile ({covered.sum():,})"),
                       Patch(color=ORANGE, label=f"outside every tile ({(~covered).sum():,})")],
              loc="lower left", bbox_to_anchor=(0, 0.12), fontsize=8, frameon=True, facecolor=SURF, edgecolor="none", framealpha=1)
    ax.set_title("NAIP 2021 tiles (60 cm) and observations")
    map_axes(ax, fl)
    cap = (f"{covered.mean() * 100:.2f}% of observations fall inside a tile footprint (footprints shrunk 80 m inward, so "
           f"a crop never runs off a tile edge). Median {int(per_tile.median()):,} observations per used tile, "
           f"max {int(per_tile.max()):,}; {len(per_tile):,} of {len(tiles):,} tiles hold at least one.")
    return fig, cap


def load_bioclim(c):
    d = Path(c["base"], "data/rasters/bioclim_current/wc_30s_current")
    files = {}
    for f in d.rglob("*.tif"):
        m = re.search(r"bio_?(\d+)\.tif$", f.name, re.I)
        if m:
            files[int(m.group(1))] = f
    if not files:
        raise FileNotFoundError(f"no bio_*.tif in {d}")
    minx, miny, maxx, maxy = c["fl_wgs"].total_bounds
    layers, transform = [], None
    for k in sorted(files):
        with rasterio.open(files[k]) as src:
            r0, c0 = src.index(minx, maxy)
            r1, c1 = src.index(maxx, miny)
            win = Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
            a = src.read(1, window=win, masked=True, boundless=True).astype("float64").filled(np.nan)
            if transform is None:
                transform = src.window_transform(win)
        layers.append(a)
    stack = np.stack(layers)
    inside = geometry_mask(c["fl_wgs"].geometry, out_shape=stack.shape[1:], transform=transform, invert=True)
    ok = inside & np.isfinite(stack).all(0)
    bg = stack[:, ok].T
    occ = c["occ"]
    col = np.floor((occ["decimalLongitude"].values - transform.c) / transform.a).astype(int)
    row = np.floor((occ["decimalLatitude"].values - transform.f) / transform.e).astype(int)
    inb = (row >= 0) & (row < stack.shape[1]) & (col >= 0) & (col < stack.shape[2])
    ob = np.full((len(occ), len(files)), np.nan)
    ob[inb] = stack[:, row[inb], col[inb]].T
    ob = ob[np.isfinite(ob).all(1)]
    return sorted(files), bg, ob


def fig_climate(c):
    keys, bg, ob = load_bioclim(c)
    k1, k12 = keys.index(1), keys.index(12)
    mu, sd = bg.mean(0), bg.std(0)
    sd[sd == 0] = 1
    _, S, Vt = np.linalg.svd((bg - mu) / sd, full_matrices=False)
    ve = S ** 2 / (S ** 2).sum() * 100
    pbg, pob = ((bg - mu) / sd) @ Vt[:2].T, ((ob - mu) / sd) @ Vt[:2].T
    tab = pd.DataFrame({"variable": [f"BIO{k}" for k in keys], "name": [BIO_NAMES.get(k, "") for k in keys],
                        "Florida land mean": mu, "observation mean": ob.mean(0),
                        "difference (SD units)": (ob.mean(0) - mu) / sd,
                        "PC1 loading": Vt[0], "PC2 loading": Vt[1]})
    save_table(c, "climate_space", tab)
    c["tables"]["climate"] = tab.sort_values("difference (SD units)", key=np.abs, ascending=False).head(6)

    pairs = [((bg[:, k1], bg[:, k12]), (ob[:, k1], ob[:, k12]), "Annual mean temperature (°C)", "Annual precipitation (mm)"),
             ((pbg[:, 0], pbg[:, 1]), (pob[:, 0], pob[:, 1]), f"PC1 ({ve[0]:.0f}% of variance)", f"PC2 ({ve[1]:.0f}%)")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.4))
    for r, (b, o, xl, yl) in enumerate(pairs):
        xs = np.concatenate([b[0], o[0]]); ys = np.concatenate([b[1], o[1]])
        xe = np.linspace(*np.percentile(xs, [0.2, 99.8]), 61)
        ye = np.linspace(*np.percentile(ys, [0.2, 99.8]), 61)
        Hb, _, _ = np.histogram2d(b[0], b[1], bins=[xe, ye])
        Ho, _, _ = np.histogram2d(o[0], o[1], bins=[xe, ye])
        ext = [xe[0], xe[-1], ye[0], ye[-1]]
        for ax, H, title, lab in [(axs[r, 0], Hb, "Florida land (30\" cells)", "cells"),
                                  (axs[r, 1], Ho, "Observations", "observations")]:
            im = ax.imshow(np.ma.masked_equal(H.T, 0), origin="lower", extent=ext, aspect="auto", cmap=SEQ, norm=LogNorm())
            cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label(f"{lab} (log)"); cb.outline.set_visible(False)
            ax.set_title(title)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log2((Ho / Ho.sum()) / (Hb / Hb.sum()))
        ratio[(Hb < 5) | ~np.isfinite(ratio)] = np.nan
        ratio[(Hb >= 5) & (Ho == 0)] = -4
        im = axs[r, 2].imshow(np.ma.masked_invalid(np.clip(ratio, -4, 4)).T, origin="lower", extent=ext, aspect="auto",
                              cmap=DIV.reversed(), norm=TwoSlopeNorm(0, -4, 4))
        cb = fig.colorbar(im, ax=axs[r, 2], pad=0.01, ticks=[-4, -2, 0, 2, 4])
        cb.ax.set_yticklabels(["1/16×", "1/4×", "even", "4×", "16×"]); cb.outline.set_visible(False)
        cb.set_label("observations relative to land area")
        axs[r, 2].set_title("Over- (blue) and under-sampled (red) climates")
        for ax in axs[r]:
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(False)
    fig.tight_layout()
    top = c["tables"]["climate"].iloc[0]
    cap = (f"{len(bg):,} Florida land cells vs {len(ob):,} observations with BioClim values. Right column: share of "
           f"observations in each climate bin divided by its share of Florida's land (cells with fewer than 5 land cells "
           f"left blank). Largest shift: {top['variable']} ({top['name']}), {top['difference (SD units)']:+.2f} SD. "
           f"PCA is fitted on the land cells with all 19 variables standardised.")
    return fig, cap


def fig_ecoregions(c):
    occ, fl = c["occ"], c["fl"]
    eco = gpd.read_file(Path(c["base"], "data/shpfiles/ecoregions/fl/fl_eco_l3.shp")).to_crs(EQ)
    name = next((n for n in eco.columns if n.upper() in ("US_L3NAME", "L3_NAME", "NA_L3NAME")), None)
    name = name or next(n for n in eco.columns if n != "geometry" and eco[n].dtype == object)
    eco = eco[[name, "geometry"]].dissolve(by=name).reset_index()
    eco["area_km2"] = eco.area / 1e6
    j = gpd.sjoin(points(c), eco[[name, "geometry"]], predicate="within", how="inner")
    j = j[~j.index.duplicated()]
    sp = occ["species"].values
    g = pd.DataFrame({"eco": j[name].values, "sp": sp[j.index.values]})
    stats = g.groupby("eco").agg(obs=("sp", "size"), species=("sp", "nunique")).reset_index()
    tab = eco[[name, "area_km2"]].rename(columns={name: "ecoregion"}).merge(stats, left_on="ecoregion", right_on="eco", how="left").drop(columns="eco")
    tab = tab.fillna(0)
    tab["obs"] = tab["obs"].astype(int); tab["species"] = tab["species"].astype(int)
    tab["obs per 100 km²"] = tab["obs"] / tab["area_km2"] * 100
    tab["% of area"] = tab["area_km2"] / tab["area_km2"].sum() * 100
    tab["% of obs"] = tab["obs"] / max(1, tab["obs"].sum()) * 100
    tab = tab.sort_values("obs per 100 km²", ascending=False).reset_index(drop=True)
    save_table(c, "ecoregions", tab)
    c["tables"]["eco"] = tab

    colors = {e: CAT[i % len(CAT)] for i, e in enumerate(tab["ecoregion"])}
    fig = plt.figure(figsize=(14, 6))
    gsp = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
    ax = fig.add_subplot(gsp[:, 0])
    eco.assign(col=eco[name].map(colors)).plot(ax=ax, color=eco[name].map(colors), edgecolor=SURF, lw=1)
    ax.legend(handles=[Patch(color=colors[e], label=e) for e in tab["ecoregion"]], loc="lower left", bbox_to_anchor=(0, 0.12), fontsize=8, frameon=True, facecolor=SURF, edgecolor="none", framealpha=1)
    ax.set_title("EPA Level III ecoregions")
    map_axes(ax, fl)
    a1 = fig.add_subplot(gsp[0, 1]); a2 = fig.add_subplot(gsp[1, 1])
    labels = [e if len(e) < 34 else e[:32] + "…" for e in tab["ecoregion"]]
    hbar(a1, labels, tab["obs per 100 km²"].values, fmt="{:,.0f}")
    a1.set_title("Observations per 100 km²")
    hbar(a2, labels, tab["species"].values, color=AQUA)
    a2.set_title("Species recorded")
    fig.tight_layout()
    cap = "Sampling effort normalised by area. Compare '% of obs' with '% of area' in the table to see which regions are over-represented."
    return fig, cap


def fig_uncertainty(c):
    u = c["occ"].get("coordinateUncertaintyInMeters")
    if u is None:
        raise KeyError("coordinateUncertaintyInMeters not in the CSV")
    miss = int(u.isna().sum())
    u = u.dropna().values
    half_crop = CROP_PX * 0.6 / 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.hist(u, bins=np.arange(0, 125, 5), color=BLUE, edgecolor=SURF, lw=1)
    a1.set_xlabel("coordinate uncertainty (m)"); a1.set_ylabel("observations")
    a1.set_title("Coordinate uncertainty")
    s = np.sort(u)
    a2.plot(s, np.arange(1, len(s) + 1) / len(s) * 100, color=BLUE, lw=2)
    a2.set_xlim(0, 120); a2.set_ylim(0, 100)
    a2.set_xlabel("coordinate uncertainty (m)"); a2.set_ylabel("% of observations at or below")
    a2.set_title("Cumulative")
    for ax in (a1, a2):
        ax.axvline(half_crop, color=INK2, lw=1, ls=(0, (4, 3)))
    a2.text(half_crop + 1.5, 8, f"half a {CROP_PX}-px crop\nat 60 cm ≈ {half_crop:.0f} m", fontsize=8, color=INK2)
    fig.tight_layout()
    pct = lambda t: (u <= t).mean() * 100
    cap = (f"{pct(10):.0f}% of records are within 10 m, {pct(30):.0f}% within 30 m, and {pct(half_crop):.0f}% within "
           f"{half_crop:.0f} m (the plant is then guaranteed to be inside its crop). {miss:,} records have no "
           f"uncertainty value.")
    return fig, cap


def fig_years(c):
    occ = c["occ"]
    yr = occ["year"].dropna().astype(int)
    inat = occ.loc[yr.index, "datasetKey"].eq(INAT) if "datasetKey" in occ else pd.Series(False, index=yr.index)
    t = pd.DataFrame({"year": yr, "inat": inat.values}).groupby(["year", "inat"]).size().unstack(fill_value=0)
    t = t.reindex(columns=[True, False], fill_value=0)
    years = t.index.values
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(years, t[True], color=BLUE, width=0.72, label="iNaturalist", edgecolor=SURF, lw=1)
    ax.bar(years, t[False], bottom=t[True], color=ORANGE, width=0.72, label="other sources", edgecolor=SURF, lw=1)
    tot = t.sum(axis=1)
    for x, v in zip(years, tot):
        ax.text(x, v, f"{v / 1000:,.0f}k" if v >= 10000 else f"{v:,}", ha="center", va="bottom", fontsize=8, color=INK2)
    if 2021 in years:
        ax.annotate("NAIP imagery year", xy=(2021, tot[2021] + tot.max() * 0.07), xytext=(2021, tot.max() * 1.12), ha="center",
                    fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_ylim(0, tot.max() * 1.22)
    ax.set_xticks(years); ax.grid(axis="x", visible=False)
    ax.set_ylabel("observations"); ax.set_title("Observations per year")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_table(c, "years", t.rename(columns={True: "iNaturalist", False: "other"}).reset_index())
    after = tot[tot.index >= 2021].sum() / tot.sum() * 100
    cap = (f"{after:.0f}% of observations are from 2021 or later; {t[True].sum() / tot.sum() * 100:.0f}% overall come "
           f"from iNaturalist. Most records postdate the 2021 imagery by a few years at most.")
    return fig, cap


def fig_season(c):
    occ = c["occ"]
    m = occ["month"].dropna().astype(int)
    top = occ["species"].value_counts().head(20).index
    sub = occ[occ["species"].isin(top)].dropna(subset=["month"])
    H = pd.crosstab(sub["species"], sub["month"].astype(int)).reindex(index=top, columns=range(1, 13), fill_value=0)
    Hn = H.div(H.max(axis=1), axis=0)
    fig = plt.figure(figsize=(13, 7.4))
    gsp = fig.add_gridspec(1, 2, width_ratios=[1, 1.35])
    a1 = fig.add_subplot(gsp[0])
    cnt = m.value_counts().reindex(range(1, 13), fill_value=0)
    a1.bar(range(1, 13), cnt.values, color=BLUE, width=0.72)
    a1.set_xticks(range(1, 13), MONTHS); a1.grid(axis="x", visible=False)
    a1.set_ylabel("observations"); a1.set_title("Observations per month (all years)")
    a2 = fig.add_subplot(gsp[1])
    im = a2.imshow(Hn.values, aspect="auto", cmap=SEQ, vmin=0, vmax=1)
    a2.set_xticks(range(12), MONTHS); a2.set_yticks(range(len(top)), list(top), fontsize=8, fontstyle="italic")
    a2.grid(False)
    cb = fig.colorbar(im, ax=a2, pad=0.01, shrink=0.7); cb.set_label("share of the species' busiest month"); cb.outline.set_visible(False)
    a2.set_title("Top 20 species by month")
    fig.tight_layout()
    save_table(c, "season_top20", H.reset_index())
    pk = cnt.idxmax(); lo = cnt.idxmin()
    cap = (f"Busiest month: {pd.Timestamp(2021, pk, 1):%B}; quietest: {pd.Timestamp(2021, lo, 1):%B} "
           f"({cnt[pk] / max(1, cnt[lo]):.1f}× fewer). Peaks in individual species usually mark flowering, when "
           f"plants are most noticed and photographed.")
    return fig, cap


def fig_taxonomy(c):
    occ = c["occ"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    out = {}
    for r, rank in enumerate(["family", "genus"]):
        if rank not in occ:
            continue
        g = occ.groupby(rank).agg(obs=("species", "size"), species=("species", "nunique"))
        out[rank] = g.sort_values("obs", ascending=False).reset_index()
        a = g.sort_values("obs", ascending=False).head(15)
        hbar(axs[r, 0], a.index, a["obs"].values)
        axs[r, 0].set_title(f"Largest {'families' if rank == 'family' else 'genera'} by observations")
        b = g.sort_values("species", ascending=False).head(15)
        hbar(axs[r, 1], b.index, b["species"].values, color=AQUA)
        axs[r, 1].set_title(f"Largest {'families' if rank == 'family' else 'genera'} by species")
        if rank == "genus":
            for ax in axs[r]:
                for t in ax.get_yticklabels():
                    t.set_fontstyle("italic")
    fig.tight_layout()
    for k, v in out.items():
        save_table(c, f"taxonomy_{k}", v)
    fam = out.get("family")
    cap = (f"{occ['family'].nunique():,} families, {occ['genus'].nunique():,} genera, {occ['species'].nunique():,} species. "
           + (f"The top 15 families hold {fam['obs'].head(15).sum() / fam['obs'].sum() * 100:.0f}% of observations."
              if fam is not None else ""))
    return fig, cap


def fig_cutoff(c):
    vc = c["occ"]["species"].value_counts()
    ts = np.unique(np.round(np.logspace(0, np.log10(vc.max()), 300)).astype(int))
    n_sp = np.array([(vc >= t).sum() for t in ts])
    f_obs = np.array([vc[vc >= t].sum() for t in ts]) / vc.sum() * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.plot(ts, n_sp, color=BLUE, lw=2); a1.set_yscale("log")
    a1.set_ylabel("species kept (log)"); a1.set_title("Species kept")
    a2.plot(ts, f_obs, color=BLUE, lw=2); a2.set_ylim(0, 102)
    a2.set_ylabel("% of observations kept"); a2.set_title("Observations kept")
    rows = []
    for t in [50, 100, 200, 500, 1000]:
        rows.append({"cutoff (records)": t, "species kept": int((vc >= t).sum()),
                     "% of observations kept": vc[vc >= t].sum() / vc.sum() * 100})
    for ax, yv in [(a1, n_sp), (a2, f_obs)]:
        ax.set_xscale("log"); ax.set_xlabel("minimum records per species")
        for t in (200, 500):
            k = int((vc >= t).sum()) if ax is a1 else vc[vc >= t].sum() / vc.sum() * 100
            ax.axvline(t, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.plot([t], [k], "o", ms=7, color=BLUE, mec=SURF, mew=2)
            ax.annotate(f"{t}: {k:,}" if ax is a1 else f"{t}: {k:.0f}%", (t, k), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, color=INK)
    fig.tight_layout()
    tab = pd.DataFrame(rows)
    save_table(c, "cutoff", tab)
    c["tables"]["cutoff"] = tab
    cap = ("Counted on the raw records (vascular plants), before Build_Data removes duplicates, records outside NAIP "
           "and so on, so the final numbers are slightly lower. The label summary has the post-filter counts.")
    return fig, cap


def naip_index(naip_dir, cache):
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    files = sorted(Path(naip_dir).rglob("*.tif"))
    log(f"  indexing {len(files):,} NAIP tiles (cached afterwards)")
    for f in files:
        try:
            with rasterio.open(f) as s:
                b = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
            rows.append({"path": str(f), "minx": b[0], "miny": b[1], "maxx": b[2], "maxy": b[3]})
        except Exception as e:
            log(f"  skip {f.name}: {e}")
    idx = pd.DataFrame(rows)
    idx.to_csv(cache, index=False)
    return idx


def read_crop(path, lon, lat, size=CROP_PX):
    with rasterio.open(path) as s:
        x, y = Transformer.from_crs("EPSG:4326", s.crs.to_wkt(), always_xy=True).transform(lon, lat)
        r, col = s.index(x, y)
        r0, c0 = r - size // 2, col - size // 2
        if r0 < 0 or c0 < 0 or r0 + size > s.height or c0 + size > s.width or s.count < 3:
            return None
        a = s.read([1, 2, 3], window=Window(c0, r0, size, size))
    if (a == 0).all(0).mean() > 0.02:
        return None
    rgb = a.transpose(1, 2, 0).astype(float)
    lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1), 0, 1)


def fig_crops(c):
    occ = c["occ"]
    idx = naip_index(c["naip_dir"], c["out"] / "tables" / "naip_tile_index.csv")
    if idx.empty:
        raise FileNotFoundError(f"no NAIP tiles in {c['naip_dir']}")
    rng = np.random.default_rng(0)
    n = c["n_crops"]
    found = {}
    u = occ.get("coordinateUncertaintyInMeters")
    for sp in c["crop_species"]:
        cand = occ[occ["species"] == sp]
        if cand.empty:
            log(f"  {sp}: not in the data, skipped"); continue
        if u is not None:
            for lim in (10, 30, 1e9):
                good = cand[u.loc[cand.index] <= lim]
                if len(good) >= 5 * n:
                    break
            cand = good if len(good) else cand
        cand = cand.iloc[rng.permutation(len(cand))[:5000]]
        lo, la = cand["decimalLongitude"].values[:, None], cand["decimalLatitude"].values[:, None]
        inside = ((idx.minx.values <= lo) & (idx.maxx.values >= lo) & (idx.miny.values <= la) & (idx.maxy.values >= la)).any(1)
        cand = cand[inside].iloc[:400]
        crops, used = [], set()
        for _, r in cand.iterrows():
            lon, lat = r["decimalLongitude"], r["decimalLatitude"]
            m = (idx.minx <= lon) & (idx.maxx >= lon) & (idx.miny <= lat) & (idx.maxy >= lat)
            for p in idx.path[m]:
                if p in used:
                    continue
                img = read_crop(p, lon, lat)
                if img is not None:
                    crops.append((img, r)); used.add(p); break
            if len(crops) == n:
                break
        if crops:
            found[sp] = crops
    if not found:
        raise RuntimeError("no crops could be read")
    rows = len(found)
    fig, axs = plt.subplots(rows, n, figsize=(2.7 * n, 2.95 * rows), squeeze=False)
    for i, (sp, crops) in enumerate(found.items()):
        for j in range(n):
            ax = axs[i, j]; ax.set_axis_off()
            if j >= len(crops):
                continue
            img, r = crops[j]
            ax.imshow(img, interpolation="nearest")
            h = img.shape[0] / 2
            ax.plot([h], [h], "o", ms=9, mfc="none", mec="white", mew=1.6)
            yr = "" if pd.isna(r.get("year")) else f"{int(r['year'])}, "
            unc = r.get("coordinateUncertaintyInMeters")
            ax.text(0.02, 0.02, f"{yr}±{unc:.0f} m" if pd.notna(unc) else yr, transform=ax.transAxes,
                    fontsize=7, color="white", va="bottom")
            if j == 0:
                common = DEFAULT_CROPS.get(sp, "")
                ax.set_title(sp + (f"  ({common})" if common else ""), fontsize=10, loc="left", fontstyle="italic")
    fig.tight_layout()
    cap = (f"{CROP_PX}×{CROP_PX}-pixel windows of NAIP 2021 (true colour, 60 cm, so about "
           f"{CROP_PX * 0.6:.0f} m across) centred on observations with low coordinate uncertainty; the ring marks the "
           f"recorded point. One crop per tile, chosen at random (seed 0). Lower-left: year and coordinate uncertainty.")
    return fig, cap


FIGS = [("density", "Where the observations are", fig_density),
        ("split", "Spatial train/test split", fig_split),
        ("naip", "Imagery coverage", fig_naip),
        ("climate", "Climate space and sampling bias", fig_climate),
        ("ecoregions", "Sampling effort by ecoregion", fig_ecoregions),
        ("uncertainty", "Location accuracy", fig_uncertainty),
        ("years", "When: years", fig_years),
        ("season", "When: season", fig_season),
        ("taxonomy", "Taxonomy", fig_taxonomy),
        ("cutoff", "Choosing the species cutoff", fig_cutoff),
        ("crops", "What the model sees", fig_crops)]
TABLE_AFTER = {"split": "split", "climate": "climate", "ecoregions": "eco", "cutoff": "cutoff"}

CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;background:#fcfcfb;color:#0b0b0b;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.5}
h1{font-size:24px;margin-bottom:4px}h2{font-size:18px;margin-top:40px;border-top:1px solid #e4e3df;padding-top:20px}
.sub{color:#52514e}.cap{color:#52514e;font-size:14px;max-width:880px}img{max-width:100%}
.wrap{overflow-x:auto}table{border-collapse:collapse;font-size:13px;margin:8px 0}
th,td{padding:4px 10px;border-bottom:1px solid #e4e3df;text-align:right}th:first-child,td:first-child{text-align:left}
th{color:#52514e;font-weight:600}.err{color:#a92a2a;font-size:13px;white-space:pre-wrap}
"""


def main(a):
    out = Path(a.out)
    (out / "png").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    log("loading occurrences")
    occ, counts = load_occ(a.occ, vascular=not a.all_plants)
    log(f"  {counts['raw']:,} rows, {counts['used']:,} used, {occ['species'].nunique():,} species")
    fl, fl_wgs = load_florida(a.base)
    c = dict(occ=occ, fl=fl, fl_wgs=fl_wgs, base=a.base, out=out, hex_km=a.hex_km, tables={},
             naip_dir=a.naip_dir, crop_species=a.crop_species, n_crops=a.n_crops)
    only = set(a.only.split(",")) if a.only else None
    sections = []
    for key, title, fn in FIGS:
        if only and key not in only:
            continue
        i = [k for k, _, _ in FIGS].index(key) + 1
        log(f"figure {i:02d} {key}")
        t0 = time.time()
        try:
            fig, cap = fn(c)
            fig.suptitle("")
            png = out / "png" / f"{i:02d}_{key}.png"
            fig.savefig(png, dpi=300, bbox_inches="tight")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            img = f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
            tab = table_html(c["tables"][TABLE_AFTER[key]], "{:,.2f}") if key in TABLE_AFTER and TABLE_AFTER[key] in c["tables"] else ""
            sections.append(f"<h2>{i}. {title}</h2>{img}<p class='cap'>{cap}</p>{tab}")
            log(f"  done in {time.time() - t0:.0f}s -> {png.name}")
        except Exception as e:
            plt.close("all")
            log(f"  FAILED: {e}")
            traceback.print_exc()
            sections.append(f"<h2>{i}. {title}</h2><p class='err'>Failed: {type(e).__name__}: {e}</p>")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Florida Deepbiosphere: data figures</title>
<style>{CSS}</style></head><body>
<h1>Florida Deepbiosphere: data figures</h1>
<p class="sub">{counts['used']:,} observations of {occ['species'].nunique():,} vascular plant species
({counts['raw']:,} rows in the GBIF download; {counts['raw'] - counts['used']:,} dropped for missing species,
coordinates or non-vascular class). GBIF download 0012182-261023074918325. Generated {time.strftime('%Y-%m-%d %H:%M')}.
300-dpi PNGs are in png/, tables in tables/.</p>
{''.join(sections)}
</body></html>"""
    (out / "figures_fl.html").write_text(html)
    log(f"report: {out / 'figures_fl.html'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=BASE)
    p.add_argument("--occ", default=f"{BASE}/data/occurrences/plant_2015_2025_USA_10_1_acq2026_10_28.csv")
    p.add_argument("--naip_dir", default=f"{BASE}/data/scratch/fl_060cm_2021")
    p.add_argument("--out", default=f"{BASE}/data/results/figures_fl")
    p.add_argument("--only", default="", help="comma-separated figure keys, e.g. density,years")
    p.add_argument("--hex_km", type=float, default=5.0)
    p.add_argument("--crop_species", nargs="+", default=list(DEFAULT_CROPS))
    p.add_argument("--n_crops", type=int, default=4)
    p.add_argument("--all_plants", action="store_true", help="keep non-vascular plants (Build_Data drops them)")
    main(p.parse_args())
