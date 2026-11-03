#!/usr/bin/env python
"""Summary statistics for the Florida Deepbiosphere labels.

Runs Build_Data.py's own label steps (no image extraction), in the same order:
  vascular plants -> species list (>= min_obs raw records) -> inside Florida (GADM)
  -> remove singletons / duplicates -> inside a NAIP 2021 tile -> joint labels within 256 m
     + drop species below --threshold -> 1-degree latitude train/test bands
then writes a self-contained HTML report, CSV tables and the labeled table.

Usage (inside the deepbio env):
  python label_summary.py [--threshold 200] [--out .../results/label_summary_fl]
"""
import argparse
import base64
import io
import json
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.spatial import cKDTree

import deepbiosphere.Build_Data as bd
import deepbiosphere.NAIP_Utils as naip
from deepbiosphere.Utils import paths

BASE = "/scratch/ajasapar/deepbiosphere"
VASCULAR = ['Gnetopsida', 'Liliopsida', 'Lycopodiopsida', 'Magnoliopsida',
            'Pinopsida', 'Polypodiopsida', 'Ginkgoopsida']
K_LIMIT = 2000        # hard-coded in Build_Data.add_overlapping_filter
RADIUS = 256          # metres, Build_Data's joint-label radius (res = 1.0)

# chart styling (light surface, one sequential blue)
SURF, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def table_html(df, floatfmt="{:,.1f}"):
    def fmt(v):
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v)
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in r) + "</tr>"
                   for r in df.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def own_joint_labels(daset, threshold, idCol="gbifID"):
    """Same logic as Build_Data.add_overlapping_filter, without its K=2000 cap."""
    d = daset.to_crs(naip.CRS.NAIP_CRS_1)
    xy = np.c_[d.geometry.x, d.geometry.y]
    nbrs = cKDTree(xy).query_ball_point(xy, r=RADIUS)
    specs = d.species.to_numpy()
    counts = pd.Series(np.concatenate([specs[n] for n in nbrs])).value_counts()
    keep = set(counts[counts >= threshold].index)
    d["overlapping_specs"] = [set(s for s in specs[n] if s in keep) for n in nbrs]
    d["len_overlap"] = d.overlapping_specs.map(len)
    d = d[d.species.isin(keep)]
    count_spec = pd.Series([s for ss in d.overlapping_specs for s in ss]).value_counts()
    return d, count_spec


def main(a):
    out = Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True)
    funnel = []

    # ---------------- load ----------------
    log("reading observations")
    cols = ["gbifID", "species", "genus", "family", "order", "class", "decimalLatitude",
            "decimalLongitude", "year", "month", "coordinateUncertaintyInMeters", "institutionCode"]
    raw = pd.read_csv(a.occ, sep="\t", usecols=cols, on_bad_lines="skip", low_memory=False)
    funnel.append(("GBIF download", len(raw), raw.species.nunique()))
    d = raw[raw["class"].isin(VASCULAR) & raw.species.notna()].copy()
    funnel.append(("Vascular plants with a species name", len(d), d.species.nunique()))
    species = set(json.load(open(a.species_file)))
    d = d[d.species.isin(species)]
    funnel.append((f"Species with >= {a.min_obs} raw records", len(d), d.species.nunique()))
    d = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.decimalLongitude, d.decimalLatitude),
                         crs=naip.CRS.GBIF_CRS)
    fl = naip.get_state_outline("fl").to_crs(naip.CRS.GBIF_CRS)
    d = gpd.sjoin(d, fl[["geometry"]], predicate="within").drop(columns="index_right")
    funnel.append(("Inside Florida (GADM)", len(d), d.species.nunique()))

    # ---------------- Build_Data steps ----------------
    log("remove_singletons_duplicates (Build_Data) -- the slow step")
    d = bd.remove_singletons_duplicates(d, 1.0)
    funnel.append(("After duplicate / singleton removal", len(d), d.species.nunique()))

    log("add_filenames (Build_Data): match to NAIP 2021 tiles")
    d = bd.add_filenames(d, "fl", "2021", "fl_060cm_2021")
    funnel.append(("Inside a downloaded NAIP tile", len(d), d.species.nunique()))

    log("checking density for Build_Data's K=2000 limit")
    p = d.to_crs(naip.CRS.NAIP_CRS_1)
    xy = np.c_[p.geometry.x, p.geometry.y]
    n_within = cKDTree(xy).query_ball_point(xy, r=RADIUS, return_length=True)
    max_nb = int(n_within.max())
    hot = d.iloc[int(n_within.argmax())]
    k_ok = max_nb < K_LIMIT
    log(f"max observations within {RADIUS} m of one point: {max_nb:,} "
        f"({'OK' if k_ok else 'EXCEEDS'} K={K_LIMIT}) at {hot.decimalLatitude:.4f}, {hot.decimalLongitude:.4f}")

    log("joint labels + threshold")
    if k_ok:
        d, count_spec, _, _ = bd.add_overlapping_filter(d, 1.0, a.threshold)
        count_spec = pd.Series(count_spec).sort_values(ascending=False)
    else:
        d, count_spec = own_joint_labels(d, a.threshold)
    funnel.append((f"Final: species with >= {a.threshold} joint labels", len(d), d.species.nunique()))

    log("spatial split bands (Build_Data)")
    split_ok = True
    try:
        d = bd.make_spatial_split(d, "decimalLatitude")
    except ValueError as e:
        # Build_Data crashes if a band has no test points; count the bands directly instead
        split_ok = False
        log(f"make_spatial_split failed ({e}); a band has no test points -- Build_Data would crash here")
        d = d.to_crs(naip.CRS.GBIF_CRS)
        ex = bd.KM_2_DEG * 1.5
        lat = d.decimalLatitude
        for i, b in enumerate(range(24, 31)):
            d[f"train_{i}"] = (lat < b) | (lat > b + 1)
            d[f"test_{i}"] = (lat > b + ex) & (lat < b + 1 - ex)

    # ---------------- tables ----------------
    log("computing summaries")
    d = d.to_crs(naip.CRS.GBIF_CRS)
    fun = pd.DataFrame(funnel, columns=["Step", "Observations", "Species"])
    sp = d.species.value_counts()
    sp_tab = pd.DataFrame({"species": sp.index, "observations": sp.values,
                           "joint_labels": count_spec.reindex(sp.index).fillna(0).astype(int).values})
    fam = d.groupby("family").agg(observations=("gbifID", "size"), species=("species", "nunique")) \
           .sort_values("observations", ascending=False).reset_index()
    eco = gpd.read_file(f"{paths.SHPFILES}ecoregions/fl/fl_eco_l3.shp").to_crs(naip.CRS.GBIF_CRS)
    de = gpd.sjoin(d[["gbifID", "species", "geometry"]], eco[["US_L3NAME", "geometry"]], how="left")
    de = de.drop_duplicates("gbifID")
    de["US_L3NAME"] = de.US_L3NAME.fillna("(outside ecoregion polygons)")
    eco_tab = de.groupby("US_L3NAME").agg(observations=("gbifID", "size"), species=("species", "nunique"))
    n_regions = de.groupby("species").US_L3NAME.nunique()
    single = de[de.species.isin(n_regions[n_regions == 1].index)]
    eco_tab["species_only_here"] = single.groupby("US_L3NAME").species.nunique() \
        .reindex(eco_tab.index).fillna(0).astype(int)
    eco_tab = eco_tab.sort_values("observations", ascending=False).reset_index().rename(columns={"US_L3NAME": "ecoregion"})
    bands = sorted(int(c.split("_")[1]) for c in d.columns if c.startswith("test_"))
    band_tab = pd.DataFrame([{"band": f"{24 + b}-{25 + b}°N", "train": int(d[f"train_{b}"].sum()),
                              "test": int(d[f"test_{b}"].sum()),
                              "test species": int(d.loc[d[f'test_{b}'], 'species'].nunique())} for b in bands])
    tiles = d[f"filepath_2021"].value_counts()
    yr = d.year.value_counts().sort_index()
    mo = d.month.value_counts().sort_index()
    lab = d.len_overlap

    for name, t in [("funnel", fun), ("species", sp_tab), ("families", fam), ("ecoregions", eco_tab),
                    ("split_bands", band_tab)]:
        t.to_csv(out / "tables" / f"{name}.csv", index=False)
    tiles.rename("observations").to_csv(out / "tables" / "obs_per_tile.csv")
    keep_cols = [c for c in d.columns if c in ("gbifID", "species", "genus", "family", "decimalLatitude",
                 "decimalLongitude", "year", "filepath_2021", "len_overlap")]
    lab_out = d[keep_cols].copy()
    lab_out["overlapping_specs"] = d.overlapping_specs.map(lambda s: "|".join(sorted(s)))
    lab_out.to_csv(out / "tables" / "labeled_observations.csv.gz", index=False)

    # ---------------- headline numbers ----------------
    top10 = sp.iloc[:max(1, len(sp) // 10)].sum() / sp.sum()
    heads = [
        ("Observations", f"{len(d):,}"), ("Species", f"{d.species.nunique():,}"),
        ("Genera", f"{d.genus.nunique():,}"), ("Families", f"{d.family.nunique():,}"),
        ("Labels per image (median)", f"{lab.median():.0f}"), ("NAIP tiles used", f"{len(tiles):,}"),
    ]

    # ---------------- figures ----------------
    figs = {}
    # funnel
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(fun.Step[::-1], fun.Observations[::-1], color=BLUE, height=0.55)
    for y, (v, s) in enumerate(zip(fun.Observations[::-1], fun.Species[::-1])):
        ax.text(v, y, f"  {v:,} obs · {s:,} spp", va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, fun.Observations.max() * 1.45); ax.grid(axis="y", visible=False)
    ax.set_title("How each filtering step shrinks the data"); ax.set_xlabel("observations")
    figs["funnel"] = fig_to_img(fig)

    # rank-abundance
    fig, ax = plt.subplots(figsize=(8, 3.6))
    r = np.arange(1, len(sp) + 1)
    ax.plot(r, sp.values, color=BLUE, lw=2, label="observations of the species")
    ax.plot(r, sp_tab.joint_labels.values, color="#eb6834", lw=2, label="images labeled with the species")
    ax.axhline(a.threshold, color=INK2, lw=1, ls="--"); ax.text(len(sp), a.threshold, f"threshold {a.threshold} ",
                                                               ha="right", va="bottom", color=INK2, fontsize=9)
    ax.set_yscale("log"); ax.set_xlabel("species rank"); ax.set_ylabel("count (log)")
    ax.set_title("Species frequency: a few very common species, a long tail"); ax.legend(frameon=False)
    figs["rank"] = fig_to_img(fig)

    # labels per image
    fig, ax = plt.subplots(figsize=(8, 3.2))
    cap = int(np.percentile(lab, 99))
    ax.hist(lab.clip(upper=cap), bins=range(1, cap + 2), color=BLUE, rwidth=0.85)
    ax.set_xlabel(f"species labels per observation (within {RADIUS} m; values above {cap} grouped)")
    ax.set_ylabel("observations"); ax.set_title("How many species each training image is labeled with")
    figs["labels"] = fig_to_img(fig)

    # maps
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax in axes:
        fl.boundary.plot(ax=ax, color=INK2, lw=0.6); ax.set_axis_off(); ax.set_aspect("equal")
    hb = axes[0].hexbin(d.decimalLongitude, d.decimalLatitude, gridsize=90, bins="log", cmap=SEQ, mincnt=1, lw=0)
    fig.colorbar(hb, ax=axes[0], shrink=0.6, label="observations per hexagon (log)")
    axes[0].set_title("Where the observations are")
    g = d.assign(gx=(d.decimalLongitude // 0.1) * 0.1 + 0.05, gy=(d.decimalLatitude // 0.1) * 0.1 + 0.05)
    rich = g.groupby(["gx", "gy"]).species.nunique().reset_index()
    sc = axes[1].scatter(rich.gx, rich.gy, c=rich.species, s=9, marker="s", cmap=SEQ,
                         norm=LogNorm(vmin=1, vmax=rich.species.max()), lw=0)
    fig.colorbar(sc, ax=axes[1], shrink=0.6, label="species per ~10 km cell (log)")
    axes[1].set_title("How many species are recorded per area")
    figs["maps"] = fig_to_img(fig)

    # time
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    axes[0].bar(yr.index.astype(int), yr.values, color=BLUE, width=0.7); axes[0].set_title("Observations per year")
    axes[1].bar(mo.index.astype(int), mo.values, color=BLUE, width=0.7); axes[1].set_title("Observations per month")
    axes[1].set_xticks(range(1, 13), list("JFMAMJJASOND"))
    for ax in axes:
        ax.grid(axis="x", visible=False)
    figs["time"] = fig_to_img(fig)

    # obs per tile
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(tiles.values, bins=np.logspace(0, np.log10(tiles.max() + 1), 40), color=BLUE, rwidth=0.85)
    ax.set_xscale("log"); ax.set_xlabel("observations per NAIP tile (log)"); ax.set_ylabel("tiles")
    ax.set_title("Observations per NAIP tile")
    figs["tiles"] = fig_to_img(fig)

    # ---------------- report ----------------
    q = sp.quantile([0.1, 0.5, 0.9]).astype(int)
    notes = [
        f"The top 10% of species account for <b>{top10:.0%}</b> of observations (Gini {gini(sp.values):.2f}). "
        f"Per-species observations: median {q[0.5]:,}, 10th–90th percentile {q[0.1]:,}–{q[0.9]:,}, "
        f"max {sp.iloc[0]:,} (<i>{sp.index[0]}</i>).",
        f"Each observation is labeled with every kept species seen within {RADIUS} m. "
        f"Median {lab.median():.0f}, mean {lab.mean():.1f}, 90th percentile {lab.quantile(.9):.0f}, max {lab.max():,} labels. "
        f"{(lab == 1).mean():.0%} of images carry only their own species.",
        f"Densest spot: <b>{max_nb:,}</b> observations within {RADIUS} m of one point "
        f"({hot.decimalLatitude:.4f}, {hot.decimalLongitude:.4f}). "
        + ("Below Build_Data's limit of 2,000, so the real Build_Data function was used."
           if k_ok else "<b>This is over Build_Data's hard-coded limit of 2,000, so Build_Data.py would crash here.</b> "
           "These numbers come from an equivalent calculation; K must be raised in add_overlapping_filter before the build."),
        ("Every latitude band has test points, so Build_Data's spatial split runs." if split_ok else
         "<b>At least one latitude band has no test points, so Build_Data.py's make_spatial_split would crash.</b> "
         "Band counts below were computed directly; empty bands must be skipped in Build_Data before the build."),
        "Not applied here: Build_Data's BioClim out-of-bounds filter and image-quality drops, "
        "which remove a small number of extra observations.",
    ]
    tiles_html = "".join(f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>' for k, v in heads)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Florida Label Summary</title>
<style>
body{{background:{SURF};color:{INK};font:15px/1.5 system-ui,-apple-system,sans-serif;max-width:1000px;margin:0 auto;padding:24px 16px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:32px 0 8px;border-top:1px solid {GRID};padding-top:20px}}
.sub{{color:{INK2};margin:0 0 20px}} img{{max-width:100%;height:auto}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}}
.tile{{border:1px solid {GRID};border-radius:8px;padding:12px}} .v{{font-size:24px;font-weight:600}} .k{{color:{INK2};font-size:13px}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0;width:100%}} th,td{{text-align:left;padding:4px 8px;border-bottom:1px solid {GRID}}}
th{{color:{INK2};font-weight:600}} td:not(:first-child),th:not(:first-child){{text-align:right}}
.wrap{{overflow-x:auto}} li{{margin:6px 0}}
</style></head><body>
<h1>Florida Deepbiosphere: label summary</h1>
<p class="sub">GBIF plant observations 2015–2025 (download 0012182-261023074918325), NAIP 2021 60 cm, threshold {a.threshold}, joint-label radius {RADIUS} m. Generated {time.strftime('%Y-%m-%d %H:%M')}.</p>
<div class="tiles">{tiles_html}</div>
<h2>Key points</h2><ul>{''.join(f'<li>{n}</li>' for n in notes)}</ul>
<h2>Filtering</h2>{figs['funnel']}<div class="wrap">{table_html(fun)}</div>
<h2>Species frequency</h2>{figs['rank']}
<h3>Most common species</h3><div class="wrap">{table_html(sp_tab.head(20))}</div>
<h3>Rarest kept species</h3><div class="wrap">{table_html(sp_tab.tail(10))}</div>
<h3>Largest families</h3><div class="wrap">{table_html(fam.head(15))}</div>
<h2>Labels per image</h2>{figs['labels']}
<h2>Where</h2>{figs['maps']}
<h3>By EPA Level III ecoregion</h3><div class="wrap">{table_html(eco_tab)}</div>
<h3>Spatial train/test bands (1° of latitude each)</h3><div class="wrap">{table_html(band_tab)}</div>
<h2>When</h2>{figs['time']}
<h2>Imagery</h2>{figs['tiles']}
<p>{len(tiles):,} tiles used; median {int(tiles.median()):,} observations per tile, max {tiles.max():,}.</p>
</body></html>"""
    (out / "label_summary_fl.html").write_text(html)
    log(f"report: {out / 'label_summary_fl.html'}")
    print(fun.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--occ", default=f"{BASE}/data/occurrences/plant_2015_2025_USA_10_1_acq2026_10_28.csv")
    p.add_argument("--species_file", default=f"{BASE}/data/occurrences/fl_species_min200.json")
    p.add_argument("--min_obs", type=int, default=200)
    p.add_argument("--threshold", type=int, default=200)
    p.add_argument("--out", default=f"{BASE}/data/results/label_summary_fl")
    main(p.parse_args())
