#!/usr/bin/env python
"""Find and download only the NAIP tiles that contain Florida observations.

Two steps:
  plan      Match observations to NAIP tiles (Microsoft Planetary Computer STAC),
            pick the smallest set of tiles that covers them, measure the download size,
            and write:
              <shpfiles>/naip_tiffs/fl_shpfl_<year>/fl_naip_<year>_index.shp   (what Build_Data.py reads)
              <data>/naip_plan/fl_<year>_tiles.csv                              (download list)
              <data>/occurrences/fl_species_min<N>.json                         (for Build_Data --species_file)
  download  Download the tiles in the plan to <scratch>/fl_060cm_<year>/<APFO5>/m_*.tif,
            the layout Build_Data.py expects. Resumable: finished tiles are skipped.
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
FL_BBOX = [-87.7, 24.4, -79.9, 31.05]
FL_ALBERS = "EPSG:3086"  # Florida GDL Albers, metres, statewide
VASCULAR = ['Gnetopsida', 'Liliopsida', 'Lycopodiopsida', 'Magnoliopsida',
            'Pinopsida', 'Polypodiopsida', 'Ginkgoopsida']  # same list as Build_Data.py


def gb(n):
    return n / 1e9


# ---------------------------------------------------------------- plan
def load_obs(csv, min_obs):
    cols = ["gbifID", "species", "class", "decimalLatitude", "decimalLongitude"]
    d = pd.read_csv(csv, sep="\t", usecols=cols, on_bad_lines="skip")
    n0 = len(d)
    d = d[d["class"].isin(VASCULAR) & d.species.notna()]
    counts = d.species.value_counts()
    keep = sorted(counts[counts >= min_obs].index)
    d = d[d.species.isin(keep)]
    print(f"observations: {n0:,} total -> {len(d):,} vascular, in the "
          f"{len(keep)} species with >= {min_obs} records")
    pts = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.decimalLongitude, d.decimalLatitude),
                           crs="EPSG:4326")
    return pts, keep


def stac_tiles(year, have=frozenset()):
    import pystac_client
    cat = pystac_client.Client.open(STAC_URL)
    # NAIP "year" is the program year (the .../v002/fl/<year>/ folder). Flights for one
    # program year can run into the next calendar year (Florida 2021 ran into Jan 2022),
    # so a datetime filter of <year>-01-01/<year>-12-31 silently drops those tiles.
    # Search two calendar years and keep the program year from the path instead.
    y = int(year)
    search = cat.search(collections=["naip"], bbox=FL_BBOX,
                        datetime=f"{y}-01-01/{y + 1}-12-31")
    rows = []
    seen = other_year = 0
    for it in search.items():
        seen += 1
        href = it.assets["image"].href.split("?")[0]
        try:
            state, pyear = href.split("/v002/")[1].split("/")[:2]
        except (IndexError, ValueError):
            continue
        if state.lower() != "fl":
            continue  # neighbouring Alabama / Georgia tiles
        if pyear != str(year):
            other_year += 1
            continue  # a different NAIP program year (e.g. fl/2023 flown in 2022)
        stem = os.path.basename(href)[:-4]           # m_3008701_ne_16_060_20211130
        parts = stem.split("_")
        rel = href.split(f"/{year}/", 1)[1]          # fl_060cm_2021/30087/m_...tif
        rows.append(dict(href=href, relpath=rel, resdir=rel.split("/")[0], stem=stem,
                         APFONAME=parts[1], quarter=parts[2], utm=parts[3],
                         date=parts[5] if len(parts) > 5 else "", gsd=it.properties.get("gsd"),
                         geometry=it.geometry))
    if not rows:
        sys.exit(f"No Florida NAIP items found for {year} on Planetary Computer.")
    df = pd.DataFrame(rows)
    df["geometry"] = [shape(g) for g in df.geometry]
    tiles = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    print(f"STAC: {seen:,} items in the bbox; {len(tiles):,} Florida {year} tiles; "
          f"{other_year:,} Florida tiles from other program years skipped")
    print("  acquisition year of the Florida tiles:",
          tiles.date.str[:4].value_counts().sort_index().to_dict())
    # if a quarter-quad was flown twice, keep the copy already on disk, else the latest flight
    tiles["on_disk"] = tiles.stem.isin(have)
    tiles = (tiles.sort_values(["on_disk", "date"])
                  .drop_duplicates(["APFONAME", "quarter"], keep="last"))
    return tiles.reset_index(drop=True)


def choose_tiles(pts, tiles, shrink_m):
    # Shrink footprints so any matched observation is >= shrink_m from the tile edge;
    # the 256 px crop (~77 m half-width at 60 cm) then never runs off the image.
    inner = tiles.copy()
    inner["geometry"] = tiles.to_crs(FL_ALBERS).buffer(-shrink_m).to_crs("EPSG:4326")
    inner = inner[~inner.geometry.is_empty]
    j = gpd.sjoin(pts[["gbifID", "geometry"]], inner[["stem", "geometry"]], predicate="within")
    # greedy cover: each observation goes to the busiest tile it falls in
    busy = j.stem.value_counts()
    j["busy"] = j.stem.map(busy)
    # prefer tiles already downloaded, so a re-plan doesn't swap them for overlapping new ones
    j["on_disk"] = j.stem.isin(set(tiles.stem[tiles.on_disk]))
    pick = j.sort_values(["on_disk", "busy"], ascending=False).drop_duplicates("gbifID")
    chosen = pick.stem.value_counts().rename("n_obs")
    sel = inner[inner.stem.isin(chosen.index)].merge(chosen, left_on="stem", right_index=True)
    print(f"tiles: {len(tiles):,} Florida tiles in {tiles.date.str[:4].mode().iat[0]}; "
          f"{len(sel):,} needed to cover {len(pick):,} of {len(pts):,} observations "
          f"({len(pts) - len(pick):,} fall outside imagery, e.g. offshore or on tile gaps)")
    return sel.sort_values("n_obs", ascending=False).reset_index(drop=True)


def head_sizes(hrefs, workers):
    import planetary_computer as pc

    def size(h):
        for attempt in range(3):
            try:
                r = requests.head(pc.sign(h), timeout=60)
                r.raise_for_status()
                return int(r.headers["Content-Length"])
            except Exception:
                time.sleep(2 * (attempt + 1))
        return -1
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(size, hrefs))


def plan(a):
    pts, species = load_obs(a.occ, a.min_obs)
    have = frozenset(p.stem for p in Path(a.scratch).glob(f"fl_*_{a.year}/*/*.tif"))
    print(f"tiles already on disk: {len(have):,}")
    tiles = stac_tiles(a.year, have)
    print("resolution folders:", tiles.resdir.value_counts().to_dict(),
          "| UTM zones:", tiles.utm.value_counts().to_dict())
    sel = choose_tiles(pts, tiles, a.shrink_m)
    print(f"measuring {len(sel):,} tiles ...")
    sel['bytes'] = head_sizes(sel.href.tolist(), a.workers)
    bad = (sel['bytes'] < 0).sum()
    total = sel['bytes'][sel['bytes'] > 0].sum()
    avg = total / max(1, (sel['bytes'] > 0).sum())
    est = total + bad * avg

    # outputs
    plan_dir = Path(a.data) / "naip_plan"; plan_dir.mkdir(parents=True, exist_ok=True)
    sel.drop(columns="geometry").to_csv(plan_dir / f"fl_{a.year}_tiles.csv", index=False)
    idx_dir = Path(a.shpfiles) / "naip_tiffs" / f"fl_shpfl_{a.year}"
    if idx_dir.exists():
        shutil.rmtree(idx_dir)
    idx_dir.mkdir(parents=True)
    # Build_Data.py turns FileName into "<FileName minus last _chunk>.tif", so add a dummy chunk
    idx = sel[["APFONAME", "stem", "n_obs", "geometry"]].copy()
    idx["FileName"] = idx.stem + "_0"
    idx[["APFONAME", "FileName", "n_obs", "geometry"]].to_file(idx_dir / f"fl_naip_{a.year}_index.shp")
    sp_file = Path(a.data) / "occurrences" / f"fl_species_min{a.min_obs}.json"
    sp_file.write_text(json.dumps(species, indent=0))

    free = shutil.disk_usage(a.scratch if os.path.exists(a.scratch) else a.data).free
    covered_by_top = sel.n_obs.cumsum() / sel.n_obs.sum()
    print("\n================ NAIP PLAN ================")
    print(f"year {a.year}, folders {sel.resdir.unique().tolist()}")
    print(f"tiles to download : {len(sel):,}  ({bad} could not be measured)")
    print(f"download size     : {gb(est):,.0f} GB  (avg {gb(avg):.2f} GB per tile)")
    new = sel[~sel.on_disk]
    print(f"  already on disk : {int(sel.on_disk.sum()):,} tiles; still to fetch: {len(new):,} "
          f"(~{gb(new['bytes'].clip(lower=0).sum() + (new['bytes'] < 0).sum() * avg):,.0f} GB)")
    unused = len(have - set(sel.stem))
    if unused:
        print(f"  {unused:,} tiles on disk are no longer in the plan (safe to delete later)")
    for frac in (0.5, 0.8, 0.9, 0.95):
        k = int((covered_by_top < frac).sum()) + 1
        print(f"  {int(frac*100)}% of observations need the top {k:,} tiles "
              f"(~{gb(sel['bytes'].iloc[:k].clip(lower=0).sum() + 0):,.0f} GB)")
    print(f"free space (df)   : {gb(free):,.0f} GB  -- also check your scratch quota")
    print(f"index shapefile   : {idx_dir}")
    print(f"species list      : {sp_file}  ({len(species)} species)")
    if any(not r.startswith("fl_060cm_") for r in sel.resdir.unique()):
        print("WARNING: not all tiles are 60 cm; Build_Data.py expects fl_060cm_<year>.")
    print("============================================")


# ---------------------------------------------------------------- download
def download(a):
    import planetary_computer as pc
    sel = pd.read_csv(Path(a.data) / "naip_plan" / f"fl_{a.year}_tiles.csv")
    if a.top:
        sel = sel.head(a.top)
    dest = [Path(a.scratch) / r for r in sel.relpath]
    todo = [(h, d, b) for h, d, b in zip(sel.href, dest, sel['bytes'])
            if not (d.exists() and (b <= 0 or d.stat().st_size == b))]
    need = sum(max(b, 0) for _, _, b in todo)
    free = shutil.disk_usage(Path(a.scratch).parent if not Path(a.scratch).exists() else a.scratch).free
    print(f"{len(sel) - len(todo):,} tiles already present, {len(todo):,} to fetch (~{gb(need):,.0f} GB); "
          f"{gb(free):,.0f} GB free")
    if need > a.max_gb * 1e9:
        sys.exit(f"Stopping: {gb(need):,.0f} GB is over --max_gb {a.max_gb}. "
                 f"Use --top N to fetch only the busiest N tiles, or raise --max_gb.")
    if need > free - 20e9:
        sys.exit("Stopping: not enough free space (keeping a 20 GB margin).")

    def fetch(job):
        href, d, b = job
        d.parent.mkdir(parents=True, exist_ok=True)
        part = d.with_suffix(".tif.part")
        for attempt in range(4):
            try:
                with requests.get(pc.sign(href), stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(part, "wb") as f:
                        for chunk in r.iter_content(1 << 22):
                            f.write(chunk)
                if b > 0 and part.stat().st_size != b:
                    raise IOError("size mismatch")
                part.rename(d)
                return True
            except Exception as e:
                err = e
                time.sleep(10 * (attempt + 1))
        print(f"FAILED {d.name}: {err}", flush=True)
        return False

    ok = fail = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(fetch, todo), 1):
            ok += res; fail += (not res)
            if i % 25 == 0 or i == len(todo):
                rate = i / (time.time() - t0) * 3600
                print(f"{i:,}/{len(todo):,} done ({fail} failed), ~{rate:,.0f} tiles/hour", flush=True)
    print(f"Finished: {ok:,} downloaded, {fail} failed. Re-run to retry failures.")
    if a.top:
        # keep the index in sync with what's on disk so Build_Data.py never asks for a missing tile
        idx_dir = Path(a.shpfiles) / "naip_tiffs" / f"fl_shpfl_{a.year}"
        shp = next(idx_dir.glob("*.shp"))
        idx = gpd.read_file(shp)
        have = {p.stem for p in Path(a.scratch).glob(f"fl_*_{a.year}/*/*.tif")}
        idx = idx[(idx.FileName.str.rsplit("_", n=1).str[0]).isin(have)]
        idx.to_file(shp)
        print(f"index trimmed to the {len(idx):,} tiles on disk")


if __name__ == "__main__":
    base = "/scratch/ajasapar/deepbiosphere/data"
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["plan", "download"])
    p.add_argument("--occ", default=f"{base}/occurrences/plant_2015_2025_USA_10_1_acq2026_10_28.csv")
    p.add_argument("--data", default=base)
    p.add_argument("--scratch", default=f"{base}/scratch")
    p.add_argument("--shpfiles", default=f"{base}/shpfiles")
    p.add_argument("--year", default="2021")
    p.add_argument("--min_obs", type=int, default=200, help="drop species with fewer raw records")
    p.add_argument("--shrink_m", type=float, default=80)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_gb", type=float, default=120)
    p.add_argument("--top", type=int, default=0, help="download only the N busiest tiles")
    a = p.parse_args()
    plan(a) if a.mode == "plan" else download(a)
