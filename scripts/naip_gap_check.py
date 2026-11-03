import glob, pandas as pd, geopandas as gpd
idx = gpd.read_file('data/shpfiles/naip_tiffs/fl_shpfl_2021/')
tifs = glob.glob('data/scratch/fl_060cm_2021/**/*.tif', recursive=True)
print('index rows', len(idx), 'tifs', len(tifs), idx.crs)

obs = pd.read_csv('data/occurrences/plant_2015_2025_USA_10_1_acq2026_10_28.csv', sep='\t',
                  usecols=['gbifID','decimalLatitude','decimalLongitude'])
pts = gpd.GeoDataFrame(obs, geometry=gpd.points_from_xy(obs.decimalLongitude, obs.decimalLatitude),
                       crs=4326).to_crs(idx.crs)
hit = gpd.sjoin(pts, idx, predicate='within')
miss = pts[~pts.gbifID.isin(hit.gbifID)]
print('missed', len(miss), f'{len(miss)/len(pts):.0%}')
print('west of 84W:', (miss.decimalLongitude < -84).mean().round(3))
d = gpd.sjoin_nearest(miss, idx, distance_col='d').drop_duplicates('gbifID').d
print(d.describe(percentiles=[.25,.5,.75,.9]))
print('within 80 m of a footprint:', (d <= 80).mean().round(3))
import numpy as np
cells = set(zip(np.floor(miss.decimalLongitude/0.0625).astype(int),
                np.floor(miss.decimalLatitude/0.0625).astype(int)))
print('extra quarter-quads needed (approx):', len(cells), '-> ~', round(len(cells)*0.484), 'GiB')
