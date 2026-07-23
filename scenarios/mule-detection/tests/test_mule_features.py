import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from mule_features import (  # noqa: E402
    FEATURE_COLS,
    PROVIDER_OUT_COLS,
    derive_account_features,
    run,
    to_provider_features,
)


def sample_ledger():
    return pd.DataFrame(
        [
            dict(step=1, type="PAYMENT", amount=100.0,
                 nameOrig="C10", nameDest="M1", isFraud=0),
            dict(step=480, type="TRANSFER", amount=5000.0,
                 nameOrig="C1", nameDest="C20", isFraud=1),
            dict(step=481, type="TRANSFER", amount=3000.0,
                 nameOrig="C2", nameDest="C20", isFraud=0),
            dict(step=482, type="TRANSFER", amount=2000.0,
                 nameOrig="C3", nameDest="C20", isFraud=0),
            dict(step=483, type="CASH_OUT", amount=9000.0,
                 nameOrig="C20", nameDest="C99", isFraud=1),
            dict(step=500, type="PAYMENT", amount=50.0,
                 nameOrig="C11", nameDest="M2", isFraud=0),
        ]
    )


class MuleFeatureTests(unittest.TestCase):
    def test_proxy_mule_has_burst_and_counterparties(self):
        features = derive_account_features(
            sample_ledger(), bank_index=0, n_banks=1, n_neg_sample=None
        ).set_index("account_id")

        mule = features.loc["C20"]
        self.assertEqual(mule["is_mule"], 1)
        self.assertGreaterEqual(mule["fan_in"], 3)
        self.assertGreaterEqual(mule["fan_out"], 1)
        self.assertGreater(mule["vel_24"], 0)
        self.assertGreaterEqual(mule["vel_48"], mule["vel_24"])
        self.assertGreaterEqual(mule["vel_72"], mule["vel_48"])

    def test_output_is_numeric_and_complete(self):
        features = derive_account_features(
            sample_ledger(), bank_index=0, n_banks=1, n_neg_sample=None
        )

        self.assertEqual(
            list(features.columns), ["account_id"] + FEATURE_COLS + ["is_mule"]
        )
        self.assertFalse(features[FEATURE_COLS].isna().any().any())
        self.assertLessEqual(set(features["is_mule"].unique()), {0, 1})

    def test_provider_boundary_removes_account_identifier(self):
        features = derive_account_features(
            sample_ledger(), bank_index=0, n_banks=1, n_neg_sample=None
        )
        provider = to_provider_features(features)

        self.assertEqual(list(provider.columns), PROVIDER_OUT_COLS)
        self.assertNotIn("account_id", provider.columns)
        self.assertEqual(len(provider), len(features))

    def test_written_provider_parquet_contains_no_identifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            output_dir = Path(tmp) / "output"
            input_dir.mkdir()
            sample_ledger().to_csv(input_dir / "paysim.csv", index=False)

            with patch.dict(
                "os.environ",
                {"INPUT_DIR": str(input_dir), "OUTPUT_DIR": str(output_dir)},
            ):
                run(bank_index=0, n_banks=1, n_neg_sample=None)

            provider = pd.read_parquet(output_dir / "accounts.parquet")
            self.assertEqual(list(provider.columns), PROVIDER_OUT_COLS)
            self.assertNotIn("account_id", provider.columns)

    def test_bank_partitions_are_disjoint_and_complete(self):
        all_accounts = set()
        total_rows = 0

        for bank_index in range(3):
            features = derive_account_features(
                sample_ledger(), bank_index=bank_index,
                n_banks=3, n_neg_sample=None
            )
            accounts = set(features["account_id"])
            self.assertTrue(all_accounts.isdisjoint(accounts))
            all_accounts.update(accounts)
            total_rows += len(features)

        single_bank = derive_account_features(
            sample_ledger(), bank_index=0, n_banks=1, n_neg_sample=None
        )
        self.assertEqual(total_rows, len(single_bank))
        self.assertEqual(all_accounts, set(single_bank["account_id"]))


if __name__ == "__main__":
    unittest.main()
