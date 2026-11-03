#!/usr/bin/env python
"""Patch scripts/naip_fl.py so NAIP program-year 2021 includes tiles flown in 2022.

Florida's NAIP 2021 flights ran into January 2022. stac_tiles() searched
datetime 2021-01-01/2021-12-31, so every tile flown in 2022 was dropped.

This patch:
  1. searches acquisition dates <year>-01-01 .. <year+1>-12-31 and keeps a tile only if
     its href folder is .../v002/fl/<year>/ (the program year), not by flight date;
  2. prints how many Florida tiles were found per acquisition year;
  3. when a quarter-quad was flown twice, or tiles overlap, prefers the copy already on disk;
  4. prints on-disk vs still-to-fetch in the plan summary.
The download step is already resumable, so after re-planning it only fetches the new tiles.

Usage (on BlueHive):
  python patch_naip_2022.py /scratch/ajasapar/deepbiosphere/scripts/naip_fl.py
Writes naip_fl.py.bak next to it first. Safe to rerun: exits if already patched.
"""
import py_compile
import shutil
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/naip_fl.py")
raw = p.read_bytes().decode()
s = raw.replace("\r\n", "\n")  # original was saved with Windows line endings

if "datetime=f\"{y}-01-01/{y + 1}-12-31\"" in s:
    sys.exit(f"{p} is already patched; nothing to do.")

EDITS = [
    ("search", '''def stac_tiles(year):
    import pystac_client
    cat = pystac_client.Client.open(STAC_URL)
    search = cat.search(collections=["naip"], bbox=FL_BBOX,
                        datetime=f"{year}-01-01/{year}-12-31")
    rows = []
    for it in search.items():
        href = it.assets["image"].href.split("?")[0]
        state = it.properties.get("naip:state") or href.split("/v002/")[1].split("/")[0]
        if state.lower() != "fl":
            continue  # neighbouring Alabama / Georgia tiles
''', '''def stac_tiles(year, have=frozenset()):
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
'''),
    ("dedup", '''    tiles = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    # if a quarter-quad was flown twice, keep the latest flight
    tiles = tiles.sort_values("date").drop_duplicates(["APFONAME", "quarter"], keep="last")
    return tiles.reset_index(drop=True)''', '''    tiles = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    print(f"STAC: {seen:,} items in the bbox; {len(tiles):,} Florida {year} tiles; "
          f"{other_year:,} Florida tiles from other program years skipped")
    print("  acquisition year of the Florida tiles:",
          tiles.date.str[:4].value_counts().sort_index().to_dict())
    # if a quarter-quad was flown twice, keep the copy already on disk, else the latest flight
    tiles["on_disk"] = tiles.stem.isin(have)
    tiles = (tiles.sort_values(["on_disk", "date"])
                  .drop_duplicates(["APFONAME", "quarter"], keep="last"))
    return tiles.reset_index(drop=True)'''),
    ("choose", '''    busy = j.stem.value_counts()
    j["busy"] = j.stem.map(busy)
    pick = j.sort_values("busy", ascending=False).drop_duplicates("gbifID")''', '''    busy = j.stem.value_counts()
    j["busy"] = j.stem.map(busy)
    # prefer tiles already downloaded, so a re-plan doesn't swap them for overlapping new ones
    j["on_disk"] = j.stem.isin(set(tiles.stem[tiles.on_disk]))
    pick = j.sort_values(["on_disk", "busy"], ascending=False).drop_duplicates("gbifID")'''),
    ("plan", '''    pts, species = load_obs(a.occ, a.min_obs)
    tiles = stac_tiles(a.year)''', '''    pts, species = load_obs(a.occ, a.min_obs)
    have = frozenset(p.stem for p in Path(a.scratch).glob(f"fl_*_{a.year}/*/*.tif"))
    print(f"tiles already on disk: {len(have):,}")
    tiles = stac_tiles(a.year, have)'''),
    ("summary", '''    print(f"tiles to download : {len(sel):,}  ({bad} could not be measured)")
    print(f"download size     : {gb(est):,.0f} GB  (avg {gb(avg):.2f} GB per tile)")''', '''    print(f"tiles to download : {len(sel):,}  ({bad} could not be measured)")
    print(f"download size     : {gb(est):,.0f} GB  (avg {gb(avg):.2f} GB per tile)")
    new = sel[~sel.on_disk]
    print(f"  already on disk : {int(sel.on_disk.sum()):,} tiles; still to fetch: {len(new):,} "
          f"(~{gb(new['bytes'].clip(lower=0).sum() + (new['bytes'] < 0).sum() * avg):,.0f} GB)")
    unused = len(have - set(sel.stem))
    if unused:
        print(f"  {unused:,} tiles on disk are no longer in the plan (safe to delete later)")'''),
]

missing = [name for name, old, _ in EDITS if old not in s]
if missing:
    sys.exit(f"Not patched: could not find the {', '.join(missing)} block(s) in {p}. "
             "The script differs from the version this patch was written for; "
             "send me the file and I'll patch it directly.")

for _, old, new in EDITS:
    s = s.replace(old, new)

bak = p.with_name(p.name + ".bak")
if not bak.exists():
    shutil.copy2(p, bak)
p.write_text(s)
py_compile.compile(str(p), doraise=True)
print(f"Patched {p} (backup: {bak}). Next: python {p} plan")
