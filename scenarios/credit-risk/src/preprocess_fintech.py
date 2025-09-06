import os
import zipfile
from kaggle import KaggleApi
import pandas as pd
import numpy as np

# # Get the KAGGLE_USERNAME and KAGGLE_KEY from your kaggle.json file downloaded from Kaggle.com > Account > Settings > API > Create new token
# os.environ["KAGGLE_USERNAME"] = "your_username"
# os.environ["KAGGLE_KEY"] = "your_key"

DATA_DIR = '/mnt/input/data'
OUT_DIR = '/mnt/output/preprocessed'

api = KaggleApi()
api.authenticate()

FILES_TO_GET = ['POS_CASH_balance.csv']

COMP = 'home-credit-default-risk'

# helper to download (will overwrite if exists)
def download_file(fname):
    target = os.path.join(DATA_DIR, fname)
    if os.path.exists(target):
        print(f"{fname} already exists, skipping download")
        return target
    api.competition_download_file(COMP, fname, path=DATA_DIR, force=True)
    # add .zip extension
    os.rename(os.path.join(DATA_DIR, fname), os.path.join(DATA_DIR, fname + '.zip'))
    print(f"Downloading {fname} to {DATA_DIR} ...")
    # API sometimes zips single files; if zipped create extraction logic
    zipped = target + '.zip'
    if os.path.exists(zipped):
        with zipfile.ZipFile(zipped, 'r') as z:
            z.extractall(DATA_DIR)
        os.remove(zipped)
    return target

# download
for f in FILES_TO_GET:
    download_file(f)

##########################

pos = pd.read_csv(os.path.join(DATA_DIR,'POS_CASH_balance.csv'))

agg_map = {}
if 'SK_DPD' in pos.columns:
  agg_map['SK_DPD'] = ['max','mean']
if 'SK_DPD_DEF' in pos.columns:
  agg_map['SK_DPD_DEF'] = ['max','mean']
if 'MONTHS_BALANCE' in pos.columns:
  agg_map['MONTHS_BALANCE'] = ['count']

if not agg_map:
  # fallback generic numeric aggs
  numcols = pos.select_dtypes(include=[np.number]).columns.tolist()
  for c in numcols[:5]:
    agg_map[c] = ['mean','max']

pos_agg = pos.groupby('SK_ID_CURR').agg(agg_map)
pos_agg.columns = ['_'.join(col).strip() for col in pos_agg.columns.values]
pos_agg.reset_index(inplace=True)
pos_agg.to_parquet(os.path.join(OUT_DIR,'pos_fintech_agg.parquet'), index=False)

print('pos_fintech preprocessing finished -> processed/pos_fintech_agg.parquet')