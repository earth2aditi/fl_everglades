#!/usr/bin/env python
"""Adapt the Deepbiosphere repo from California to Florida.  Safe to run more than once.

Usage:  python patch_florida.py /scratch/ajasapar/deepbiosphere/Deepbiosphere

Changes:
  NAIP_Utils.py  NAIP_CRS_1 / NAIP_CRS_2: UTM 11N / 10N (California) -> UTM 17N (Florida).
                 NAIP_CRS_1 is the projection Build_Data.py measures distances in
                 (256 m duplicate filter, test-split exclusion zones).
  Build_Data.py  make_spatial_split / generate_split_polygons default bounding box:
                 California (lon -125..-114, lat 32..42.1) -> Florida (lon -88..-79.5, lat 24..31.1).
                 With the California box, Florida (lat 24.5-31) gets zero spatial bands.
"""
import pathlib
import sys

repo = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
src = repo / "src" / "deepbiosphere"

EDITS = {
    "NAIP_Utils.py": [
        ("NAIP_CRS_1 = 'EPSG:26911'", "NAIP_CRS_1 = 'EPSG:26917'"),
        ("NAIP_CRS_2 = 'EPSG:26910'", "NAIP_CRS_2 = 'EPSG:26917'"),
    ],
    "Build_Data.py": [
        ("def generate_split_polygons(lonmin=-125,\n                            lonmax=-114,\n"
         "                            latmin=32,\n                            latmax=42.1,",
         "def generate_split_polygons(lonmin=-88,\n                            lonmax=-79.5,\n"
         "                            latmin=24,\n                            latmax=31.1,"),
        ("def make_spatial_split(daset, latCol,\n                       lonmin=-125,\n"
         "                       lonmax=-114,\n                       latmin=32,\n                       latmax=42.1,",
         "def make_spatial_split(daset, latCol,\n                       lonmin=-88,\n"
         "                       lonmax=-79.5,\n                       latmin=24,\n                       latmax=31.1,"),
    ],
}

ok = True
for fname, edits in EDITS.items():
    path = src / fname
    text = path.read_text()
    backup = path.with_name(path.name + ".ca_orig")
    if not backup.exists():
        backup.write_text(text)
    for old, new in edits:
        label = new.splitlines()[0] if "\n" not in new else new.split("\n")[0] + " ..."
        if new in text:
            print(f"[already done] {fname}: {label}")
        elif old in text:
            text = text.replace(old, new, 1)
            print(f"[patched]      {fname}: {label}")
        else:
            print(f"[NOT FOUND]    {fname}: {label}")
            ok = False
    path.write_text(text)

print("Originals saved as *.ca_orig next to each file." if ok else
      "Some edits were not found -- paste this output to Claude.")
sys.exit(0 if ok else 1)
