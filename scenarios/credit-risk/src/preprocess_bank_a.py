# 2025 DEPA Foundation
#
# This work is dedicated to the public domain under the CC0 1.0 Universal license.
# To the extent possible under law, DEPA Foundation has waived all copyright and 
# related or neighboring rights to this work. 
# CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/)
#
# This software is provided "as is", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a 
# particular purpose and noninfringement. In no event shall the authors or copyright
# holders be liable for any claim, damages or other liability, whether in an action
# of contract, tort or otherwise, arising from, out of or in connection with the
# software or the use or other dealings in the software.
#
# For more information about this framework, please visit:
# https://depa.world/training/depa_training_framework/

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

FILES_TO_GET = [
'application_train.csv',
'application_test.csv',
'previous_application.csv',
'installments_payments.csv'
]

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

###########################


app_cols = [
'SK_ID_CURR','TARGET','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY',
'CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',
'NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',
'DAYS_BIRTH','DAYS_EMPLOYED','FLAG_MOBIL','REGION_POPULATION_RELATIVE','OBS_30_CNT_SOCIAL_CIRCLE',
'OBS_60_CNT_SOCIAL_CIRCLE','DAYS_ID_PUBLISH'
]
# Get header first
available_cols = pd.read_csv(os.path.join(DATA_DIR, "application_train.csv"), nrows=0).columns

# Load with column filtering
app_train = pd.read_csv(
    os.path.join(DATA_DIR, "application_train.csv"),
    usecols=[c for c in app_cols if c in available_cols]
)

app_train = app_train.dropna()

# Feature examples
app_train['INCOME_PER_PERSON'] = app_train['AMT_INCOME_TOTAL'] / (app_train['CNT_CHILDREN'].replace({0:1}) + 1)
app_train['CREDIT_TO_INCOME'] = app_train['AMT_CREDIT'] / (app_train['AMT_INCOME_TOTAL'].replace({0:1}))

cat_cols = app_train.select_dtypes(include=['object']).columns.tolist()
app_train[['SK_ID_CURR','TARGET','INCOME_PER_PERSON','CREDIT_TO_INCOME'] + [c for c in cat_cols]].to_parquet(os.path.join(OUT_DIR, 'bank_a_app_learn.parquet'), index=False)

# previous_application aggregation (per SK_ID_CURR)
prev = pd.read_csv(os.path.join(DATA_DIR, 'previous_application.csv'))
# create a couple of useful aggs
prev_agg = prev.groupby('SK_ID_CURR').agg({
'AMT_APPLICATION': ['count','mean','sum'],
'AMT_CREDIT': ['mean','max'],
'AMT_DOWN_PAYMENT': ['mean'],
'CNT_PAYMENT': ['mean']
})
# flatten
prev_agg.columns = ['_'.join(col).strip() for col in prev_agg.columns.values]
prev_agg.reset_index(inplace=True)
prev_agg.to_parquet(os.path.join(OUT_DIR, 'bank_a_prev_agg.parquet'), index=False)

# installments_payments aggregation
inst = pd.read_csv(os.path.join(DATA_DIR, 'installments_payments.csv'))
# create basic payment discipline features
inst['PAYMENT_DIFF'] = inst['AMT_PAYMENT'] - inst['AMT_INSTALMENT']
inst['LATE_PAYMENT'] = (inst['DAYS_ENTRY_PAYMENT'] > 0).astype(int)
inst_agg = inst.groupby('SK_ID_CURR').agg({
'AMT_PAYMENT': ['sum','mean'],
'AMT_INSTALMENT': ['sum','mean'],
'PAYMENT_DIFF': ['mean'],
'LATE_PAYMENT': ['sum','mean']
})
inst_agg.columns = ['_'.join(col).strip() for col in inst_agg.columns.values]
inst_agg.reset_index(inplace=True)
inst_agg.to_parquet(os.path.join(OUT_DIR, 'bank_a_inst_agg.parquet'), index=False)

print('bank_a preprocessing finished -> processed/*.parquet')

##########################

