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

FILES_TO_GET = ['bureau.csv','bureau_balance.csv']

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

bureau = pd.read_csv(os.path.join(DATA_DIR,'bureau.csv'))
balance = pd.read_csv(os.path.join(DATA_DIR,'bureau_balance.csv'))

bureau = bureau.dropna()
balance = balance.dropna()

# aggregate bureau_balance by SK_ID_BUREAU first: e.g., proportion of months past-due
if 'STATUS' in balance.columns:
  bal_agg = balance.groupby('SK_ID_BUREAU').agg({'MONTHS_BALANCE': ['count'], 'STATUS': lambda x: (x.astype(str).str.contains('2|3|4|5')).mean()})
  bal_agg.columns = ['MONTHS_COUNT','STATUS_PAST_DUE_RATE']
  bal_agg.reset_index(inplace=True)
  bureau = bureau.merge(bal_agg, how='left', left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU')

# now aggregate bureau by SK_ID_CURR
agg_map = {}
# defensive column existence checks
for col in ['AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_OVERDUE','DAYS_CREDIT','DAYS_ENDDATE_FACT']:
  if col in bureau.columns:
    agg_map[col] = ['mean','max']

# also count number of bureau records
if 'SK_ID_BUREAU' in bureau.columns:
  bureau['BUREAU_COUNT'] = 1
  agg_map['BUREAU_COUNT'] = ['sum']

if not agg_map:
  raise RuntimeError('No expected columns found in bureau.csv; inspect file')

bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_map)
# flatten
bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
bureau_agg.reset_index(inplace=True)
bureau_agg.to_parquet(os.path.join(OUT_DIR, 'bureau_agg.parquet'), index=False)

print('bureau preprocessing finished -> processed/bureau_agg.parquet')

