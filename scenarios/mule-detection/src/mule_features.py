# 2025 DEPA Foundation
#
# This work is dedicated to the public domain under the CC0 1.0 Universal license.
# CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/)
#
# Mule Account Detection Consortium (MADC) — shared per-bank feature derivation.
#
# Each bank runs its own preprocessor container, but they all share this logic: from
# the PaySim transaction log, derive the proposal's account-level behavioural signals
# + a binary `is_mule` label, for the accounts that belong to this bank. Banks hold
# DISJOINT account populations (the horizontal-consortium model): an account is
# assigned to a bank deterministically by its id, so every bank derives the same
# partition independently without coordination.
#
# The join inside the Confidential Clean Room UNIONs the three banks' account tables.
# `account_id` is used only while deriving a bank's rows and is removed before the
# provider dataset is written.

import os
from glob import glob

import numpy as np
import pandas as pd

# PaySim: `step` is 1 hour; nameOrig is always a customer (C...), nameDest is a
# customer (C...) or a merchant (M...). Mules are customer accounts, so we only build
# features for C-accounts (merchants are never focal accounts, only counterparties).
DATASET = "ealaxi/paysim1"
USECOLS = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]

FEATURE_COLS = [
    "account_age_days",
    "vel_24", "vel_48", "vel_72",
    "fan_in", "fan_out",
    "dormancy_ratio",
]
OUT_COLS = ["account_id"] + FEATURE_COLS + ["is_mule"]
PROVIDER_OUT_COLS = FEATURE_COLS + ["is_mule"]


def _acct_bank(account_ids, n_banks):
    """Deterministic bank assignment from a C-account id ('C1231006815' -> int % n)."""
    return account_ids.str.slice(1).astype("int64") % n_banks


def _perspective(df, focal, counterparty, direction, bank_index, n_banks):
    """One account-centric view of the ledger, filtered to this bank's C-accounts."""
    f = df[[focal, "step", "amount", counterparty]].rename(
        columns={focal: "account", counterparty: "counterparty"}
    )
    f = f[f["account"].str.startswith("C")]
    f = f[_acct_bank(f["account"], n_banks) == bank_index]
    f = f.copy()
    f["direction"] = direction
    return f


def derive_account_features(df, bank_index, n_banks, n_neg_sample=50000, seed=42):
    """PaySim transactions -> one row per (this bank's) account with the 5 signals + label.

    Pure function (no I/O) so it can be unit-checked without Kaggle. See demo()."""
    max_step = int(df["step"].max())

    # Label set (global): a mule is the account that RECEIVES fraudulent money (dest of a
    # fraudulent TRANSFER) or CASHES IT OUT (orig of a fraudulent CASH_OUT). Rare positive.
    fraud = df[df["isFraud"] == 1]
    mule_ids = set(fraud.loc[fraud["type"] == "TRANSFER", "nameDest"]) | \
        set(fraud.loc[fraud["type"] == "CASH_OUT", "nameOrig"])

    # Events for this bank's accounts, from both the sending and receiving side.
    out = _perspective(df, "nameOrig", "nameDest", "out", bank_index, n_banks)
    inb = _perspective(df, "nameDest", "nameOrig", "in", bank_index, n_banks)
    ev = pd.concat([out, inb], ignore_index=True)
    if ev.empty:
        return pd.DataFrame(columns=OUT_COLS)

    g = ev.groupby("account", sort=False)
    agg = pd.DataFrame({"first_step": g["step"].min(), "last_step": g["step"].max()})

    # Account age: older accounts appear earlier in the log.
    # PaySim has no account-open date; first-seen step is used as a sample proxy.
    agg["account_age_days"] = (max_step - agg["first_step"]) / 24.0
    span = (agg["last_step"] - agg["first_step"]).clip(lower=0)

    # Counterparty fan-in / fan-out: distinct senders / receivers.
    # These are whole-period distinct counts, not per-window counts. They are
    # discriminative for mules (many unrelated sources); tighten to peak-window if needed.
    fan_in = ev[ev["direction"] == "in"].groupby("account")["counterparty"].nunique()
    fan_out = ev[ev["direction"] == "out"].groupby("account")["counterparty"].nunique()
    agg["fan_in"] = fan_in.reindex(agg.index).fillna(0).astype("int64")
    agg["fan_out"] = fan_out.reindex(agg.index).fillna(0).astype("int64")

    # Dormancy -> activity: longest inactive gap as a fraction of the active span.
    # This is a sample proxy heuristic for "sudden spike after inactivity".
    evs = ev.sort_values(["account", "step"])
    evs = evs.assign(_gap=evs.groupby("account")["step"].diff())
    max_gap = evs.groupby("account")["_gap"].max().reindex(agg.index)
    agg["dormancy_ratio"] = (max_gap / span.replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0)

    # Transaction velocity: peak rolling-window (in+out) value over 24/48/72h.
    # Hourly totals per account, then max trailing-window sum (a genuine rolling window).
    h = ev.groupby(["account", "step"], as_index=False)["amount"].sum()
    h["ts"] = pd.to_datetime(h["step"], unit="h")
    h = h.sort_values(["account", "ts"])
    for w in (24, 48, 72):
        roll = h.groupby("account", sort=False).rolling(f"{w}h", on="ts")["amount"].sum()
        vel = roll.groupby(level=0).max()
        agg[f"vel_{w}"] = vel.reindex(agg.index).fillna(0.0)

    agg["is_mule"] = agg.index.isin(mule_ids).astype("int64")
    result = agg.reset_index().rename(columns={"account": "account_id"})[OUT_COLS].copy()

    # Keep every mule; subsample the (huge) negative class so a local run is tractable.
    # PoC downsample: drop the cap for a full-population run.
    pos = result[result["is_mule"] == 1]
    neg = result[result["is_mule"] == 0]
    if n_neg_sample is not None and len(neg) > n_neg_sample:
        neg = neg.sample(n=n_neg_sample, random_state=seed)
    result = pd.concat([pos, neg], ignore_index=True)

    for c in FEATURE_COLS:
        result[c] = pd.to_numeric(result[c], errors="coerce").fillna(0.0)
    result["is_mule"] = result["is_mule"].astype("int64")
    return result


def to_provider_features(features):
    """Remove local account identifiers before writing a provider dataset."""
    return features[PROVIDER_OUT_COLS].copy()


def run(bank_index, n_banks=3, n_neg_sample=50000):
    """Download PaySim, derive this bank's account features, write accounts.parquet."""
    data_dir = os.environ.get("INPUT_DIR", "/mnt/input/data")
    out_dir = os.environ.get("OUTPUT_DIR", "/mnt/output/preprocessed")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    csvs = glob(os.path.join(data_dir, "*.csv"))
    if not csvs:
        # Each sample bank downloads PaySim into its own directory;
        # fine for a local PoC, wasteful at scale — share a read-only volume if it hurts.
        from kaggle import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {DATASET} -> {data_dir} ...")
        api.dataset_download_files(DATASET, path=data_dir, unzip=True)
        csvs = glob(os.path.join(data_dir, "*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No PaySim CSV found in {data_dir}")

    df = pd.read_csv(csvs[0], usecols=USECOLS)
    feats = derive_account_features(df, bank_index, n_banks, n_neg_sample=n_neg_sample)
    out_path = os.path.join(out_dir, "accounts.parquet")
    provider_feats = to_provider_features(feats)
    provider_feats.to_parquet(out_path, index=False)
    print(f"bank_{bank_index + 1} preprocessing -> accounts.parquet "
          f"({len(provider_feats)} accounts, "
          f"{int(provider_feats['is_mule'].sum())} mule proxies)")


def demo():
    """Self-check on a tiny synthetic ledger — no Kaggle/network needed."""
    rows = [
        dict(step=1,   type="PAYMENT",  amount=100.0,  nameOrig="C10", nameDest="M1",  isFraud=0),
        dict(step=500, type="TRANSFER", amount=5000.0, nameOrig="C1",  nameDest="C20", isFraud=1),
        dict(step=500, type="TRANSFER", amount=3000.0, nameOrig="C2",  nameDest="C20", isFraud=0),
        dict(step=501, type="TRANSFER", amount=2000.0, nameOrig="C3",  nameDest="C20", isFraud=0),
        dict(step=502, type="CASH_OUT", amount=9000.0, nameOrig="C20", nameDest="C99", isFraud=1),
    ]
    df = pd.DataFrame(rows)
    feats = derive_account_features(df, bank_index=0, n_banks=1, n_neg_sample=None)

    assert list(feats.columns) == OUT_COLS, feats.columns.tolist()
    assert set(feats["is_mule"].unique()) <= {0, 1}
    assert not feats[FEATURE_COLS].isna().any().any(), "NaN leaked into features"
    assert {0, 1} <= set(feats["is_mule"].unique()), "need both classes for training"

    c20 = feats.set_index("account_id").loc["C20"]
    assert c20["is_mule"] == 1, "C20 (fraud dest + cashout orig) must be a mule"
    assert c20["fan_in"] >= 3, "C20 received from 3 distinct senders"
    assert c20["vel_24"] > 0, "C20 burst velocity must be positive"
    print(f"mule_features self-check passed: {len(feats)} accounts, "
          f"{int(feats['is_mule'].sum())} mule(s)")


if __name__ == "__main__":
    demo()
