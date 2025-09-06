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

FILES_TO_GET = ['credit_card_balance.csv']

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

cc = pd.read_csv(os.path.join(DATA_DIR,'credit_card_balance.csv'))

# important derived features: utilization ratio, mean balance, max days past due
cols_for_agg = {}
if 'AMT_BALANCE' in cc.columns:
  cols_for_agg['AMT_BALANCE'] = ['mean','max']
if 'AMT_CREDIT_LIMIT_ACTUAL' in cc.columns:
  cols_for_agg['AMT_CREDIT_LIMIT_ACTUAL'] = ['mean','max']
if 'AMT_DRAWINGS_CURRENT' in cc.columns:
  cols_for_agg['AMT_DRAWINGS_CURRENT'] = ['mean','max']
if 'SK_DPD' in cc.columns:
  cols_for_agg['SK_DPD'] = ['max','mean']
if 'SK_DPD_DEF' in cc.columns:
  cols_for_agg['SK_DPD_DEF'] = ['max','mean']

if not cols_for_agg:
  numcols = cc.select_dtypes(include=[np.number]).columns.tolist()
  for c in numcols[:5]:
    cols_for_agg[c] = ['mean','max']

cc_agg = cc.groupby('SK_ID_CURR').agg(cols_for_agg)
cc_agg.columns = ['_'.join(col).strip() for col in cc_agg.columns.values]
cc_agg.reset_index(inplace=True)
cc_agg.to_parquet(os.path.join(OUT_DIR,'bank_b_cc_agg.parquet'), index=False)

print('bank_b preprocessing finished -> processed/bank_b_cc_agg.parquet')