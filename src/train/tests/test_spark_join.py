import os
import tempfile
import unittest
from pathlib import Path

from pyspark.sql import SparkSession

from pytrain.join import SparkJoin


os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")


class SparkJoinMultiplicityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("spark-join-multiplicity-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def prepare_union(self, joined_dataset_config):
        columns = ["feature", "is_mule"]
        bank_1 = self.spark.createDataFrame([(1.0, 0)], columns)
        bank_2 = self.spark.createDataFrame([(1.0, 0)], columns)
        dataset_info = [(bank_1, "bank_1"), (bank_2, "bank_2")]
        query = "SELECT * FROM bank_1 UNION ALL SELECT * FROM bank_2"

        with tempfile.TemporaryDirectory() as tmp:
            result = SparkJoin().prepare_joined_dataset(
                self.spark,
                dataset_info,
                joined_dataset_config,
                query,
                str(Path(tmp) / "joined.parquet"),
                "parquet",
            )
            count = result.count()
            result.unpersist()
            return count

    def test_default_keeps_legacy_deduplication(self):
        self.assertEqual(self.prepare_union({}), 1)

    def test_union_all_can_preserve_identical_rows(self):
        self.assertEqual(self.prepare_union({"deduplicate_rows": False}), 2)

    def test_deduplicate_rows_requires_boolean(self):
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            self.prepare_union({"deduplicate_rows": "false"})


if __name__ == "__main__":
    unittest.main()
