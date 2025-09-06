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
import json
import argparse
import shutil
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType
from pyspark.sql.functions import col, column

from .task_base import TaskBase


# class SparkJoin(TaskBase):
    
#     def load_tdp_list(self, config, debug=True):
#         """
#         Extract List of TDP configurations for data joining process from query config
#         """
#         return config["datasets"]

#     def load_joined_dataset_config(cself, config, debug=True):
#         """
#         Extract List of query join configurations for data joining process from query config.
#         """
#         return config["joined_dataset"]

#     def get_name(self, tdp_config_list, debug=True):
#         """
#         Extract list of names of all TDP's from query config
#         """
#         name_list = []
#         for c in tdp_config_list:
#             name_list.append(c["name"])

#         return name_list

#     def create_spark_context(self, tdp_config_list, debug=True):
#         """
#         Create a spark session with app context auto generated from TDP names
#         """
#         name_list = self.get_name(tdp_config_list)
#         context = ""
#         for c in name_list:
#             context = context + c + "_"

#         return SparkSession.builder.appName(context).getOrCreate()

#     def generate_query(self, tdp_config_list, joined_dataset_config, debug=True):
#         """
#         Extract the query logic from the query config. Current implementation extracts query from query config.
#         """
#         # Very light validation to disallow multiple statements / dangerous tokens
#         q = joined_dataset_config["joining_query"]
#         if not isinstance(q, str):
#             raise ValueError("joining_query must be a string")
#         # Disallow semicolons and common DDL/DML keywords
#         banned = [";", "DROP ", "TRUNCATE ", "ALTER ", "CREATE ", "INSERT ", "UPDATE ", "DELETE "]
#         uq = q.upper()
#         if any(b in uq for b in banned):
#             raise ValueError("Disallowed SQL construct in joining_query")
#         return q

#     def dropDupeDfCols(self, df, debug=True):
#         """
#         Drops Duplicate Columns from the dataframe passed
#         """

#         newcols = []
#         dupcols = []

#         for i in range(len(df.columns)):
#             if df.columns[i] not in newcols:
#                 newcols.append(df.columns[i])
#             else:
#                 dupcols.append(i)

#         df = df.toDF(*[str(i) for i in range(len(df.columns))])
#         for dupcol in dupcols:
#             df = df.drop(str(dupcol))

#         return df.toDF(*newcols)

#     def dp_load_data(self, spark, input_folder, data_file, load=True, debug=True):
#         """
#         Generic Data Loading Function at Data Provider
#         """

#         if load:
#             input_file = input_folder + data_file

#             data_loaded = spark.read.csv(
#                 input_file, header=True, inferSchema=True, mode="DROPMALFORMED"
#             )

#         return data_loaded

#     def create_view(self, dataset_name, view_name):
#         """
#         Create the temp query-able views for spark query processing
#         """
#         return dataset_name.createOrReplaceTempView(view_name)

#     def ccr_prepare_joined_dataset(self, dataset_info, query, save_path, debug=True):
#         """
#         In CCR/Sandbox we are creating the joined dataset anon from the three TDP and joined data configurations . This function abstracts this and returns the dataet.
#         """
#         # Create the temp query-able views
#         for c in dataset_info:
#             self.create_view(c[0], c[1])
#         # Query Execution
#         retun_dataset = spark.sql(query).cache()
#         retun_dataset = retun_dataset.drop("_c0")
#         # Save the dataset
#         retun_dataset.toPandas().to_csv(save_path)

#         return retun_dataset

#     def ccr_prepare_joined_dataset_full(
#         self, spark, dataset_info, joined_dataset_config, query, save_path, debug=True
#     ):
#         """
#         In CCR/Sandbox we are creating the joined dataset anon from the three TDPs. This function abstracts this and returns the dataet.
#         """
#         # Create the temp query-able views
#         for c in dataset_info:
#             self.create_view(c[0], c[1])

#         # Query Execution
#         return_dataset = spark.sql(query).cache()
#         return_dataset = return_dataset.drop("_c0")
#         return_dataset_step1 = return_dataset.dropDuplicates()
#         return_dataset_step2 = self.dropDupeDfCols(return_dataset_step1)
#         drop_columns = joined_dataset_config["drop_columns"]
#         return_dataset_step3 = return_dataset_step2.drop(*drop_columns)
#         return_dataset_final = return_dataset_step3

#         return_dataset_final.toPandas().to_csv(save_path)

#         return return_dataset_final

#     def ccr_create_joined_dataset_wo_identifiers(
#         self, joined_dataset, joined_dataset_config, modelfile, debug=True
#     ):
#         identifiers = joined_dataset_config["identifiers"]
#         return_dataset = joined_dataset.drop(*identifiers)
#         # Modeling dataset
#         return_dataset.toPandas().to_csv(modelfile)

#         return return_dataset

#     def generate_data_info(self, spark, tdp_config_list):
#         """
#         Extracts the list of loaded data set along with its alias from TDP config.
#         Required for Spark view creation-create_view
#         """
#         lis = []
#         for c in tdp_config_list:
#             l = []
#             l.append(self.dp_load_data(spark, c["mount_path"], c["file"]))
#             l.append(c["name"])
#             lis.append(l)
#         return lis

#     def generate_base_query_dataset(self, dataset_info):
#         """
#         This will generate the base joined dataset for ccr_prepare_joined_dataset_full function.
#         """
#         # sandbox_icmr_cowin_index_linked_anon
#         file_str = "sandbox_"
#         for c in dataset_info:
#             file_str = file_str + c[1] + "_"
#         file_str = file_str + "linked_anon.csv"

#         return file_str

#     def execute(self, config):
#         """
#         Final Execution Function
#         """
#         tdp_config_list = self.load_tdp_list(config)
#         joined_dataset_config = self.load_joined_dataset_config(config)
#         spark = self.create_spark_context(
#             tdp_config_list
#         )  # currently treated as a global instance but can be converted into a specific instance for multiple pipelines
#         spark.sparkContext.setLogLevel("ERROR")
#         query = self.generate_query(tdp_config_list, joined_dataset_config)
#         dataset_info = self.generate_data_info(spark, tdp_config_list)
#         # sandbox_joined_anon_simplified=ccr_prepare_joined_dataset_full(dataset_info,query,joined_dataset_config["save_path"],debug=True)
#         save_path = joined_dataset_config["joined_dataset"]
#         sandbox_joined_anon_simplified = self.ccr_prepare_joined_dataset_full(
#             spark, dataset_info, joined_dataset_config, query, save_path, debug=False
#         )
#         dataset_names = [dataset["name"] for dataset in config["datasets"]]
#         print(f"Joined datasets {dataset_names} in {save_path}")
#         sandbox_joined_without_key_identifiers = (
#             self.ccr_create_joined_dataset_wo_identifiers(
#                 sandbox_joined_anon_simplified,
#                 joined_dataset_config,
#                 save_path,
#                 True,
#             )
#         )


class SparkJoin(TaskBase):
    """Utility class for loading, joining, and exporting Spark datasets based on config."""

    # -------------------------------------------------------------------------
    # Config Loaders
    # -------------------------------------------------------------------------

    def load_tdp_list(self, config):
        """Extract list of dataset (TDP) configurations."""
        return config.get("datasets", [])

    def load_joined_dataset_config(self, config):
        """Extract joined dataset configuration."""
        return config.get("joined_dataset", {})

    # -------------------------------------------------------------------------
    # Spark Context
    # -------------------------------------------------------------------------

    def get_dataset_names(self, tdp_config_list):
        """Return dataset names from config."""
        return [c["name"] for c in tdp_config_list]

    def create_spark_session(self, tdp_config_list):
        """Create Spark session with app name based on dataset names."""
        app_name = "_".join(self.get_dataset_names(tdp_config_list))
        return SparkSession.builder.appName(app_name).getOrCreate()

    # -------------------------------------------------------------------------
    # Query Handling
    # -------------------------------------------------------------------------

    def validate_and_get_query(self, joined_dataset_config):
        """Validate and return query string from config."""
        query = joined_dataset_config.get("joining_query")
        if not isinstance(query, str):
            raise ValueError("joining_query must be a string")

        banned_tokens = [";", "DROP ", "TRUNCATE ", "ALTER ", "CREATE ",
                         "INSERT ", "UPDATE ", "DELETE "]
        if any(token in query.upper() for token in banned_tokens):
            raise ValueError("Disallowed SQL construct in joining_query")

        return query

    # -------------------------------------------------------------------------
    # Data I/O
    # -------------------------------------------------------------------------

    def read_data(
        self,
        spark,
        path,
        fmt 
    ):
        """Read dataset from given path with specified format."""
        fmt = fmt.lower()
        if fmt == "csv":
            return spark.read.csv(path, header=True, inferSchema=True, mode="DROPMALFORMED")
        elif fmt == "parquet":
            return spark.read.parquet(path)
        elif fmt == "hdf5":
            # Spark does not natively support HDF5 — fallback to pandas then parallelize
            import pandas as pd
            pdf = pd.read_hdf(path)
            return spark.createDataFrame(pdf)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def write_data(
        self,
        df,
        path,
        fmt
    ) -> None:
        """Write dataset to given path with specified format."""
        fmt = fmt.lower()
        if fmt == "csv":
            df.toPandas().to_csv(path, index=False)
        elif fmt == "parquet":
            df.write.mode("overwrite").parquet(path)
        elif fmt == "hdf5":
            import pandas as pd
            df.toPandas().to_hdf(path, key="data", mode="w")
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    # -------------------------------------------------------------------------
    # Data Processing
    # -------------------------------------------------------------------------

    def drop_duplicate_columns(self, df):
        """Drop duplicate columns from DataFrame."""
        seen, duplicates = [], []
        for idx, col_name in enumerate(df.columns):
            if col_name in seen:
                duplicates.append(idx)
            else:
                seen.append(col_name)

        df = df.toDF(*map(str, range(len(df.columns))))
        for dup in duplicates:
            df = df.drop(str(dup))
        return df.toDF(*seen)

    def create_temp_views(self, dataset_info):
        """Register each dataset as a temporary SQL view."""
        for df, alias in dataset_info:
            df.createOrReplaceTempView(alias)

    def prepare_joined_dataset(
        self,
        spark,
        dataset_info,
        joined_dataset_config,
        query,
        save_path,
        fmt 
    ):
        """
        Create joined dataset by running SQL query and applying cleaning rules.
        - Drops `_c0` (index col from CSVs).
        - Drops duplicates.
        - Removes explicitly configured columns.
        - Saves to disk in requested format.
        """
        self.create_temp_views(dataset_info)

        df = spark.sql(query).cache()
        if "_c0" in df.columns:
            df = df.drop("_c0")

        df = df.dropDuplicates()
        df = self.drop_duplicate_columns(df)

        drop_cols = joined_dataset_config.get("drop_columns", [])
        if drop_cols:
            df = df.drop(*drop_cols)

        self.write_data(df, save_path, fmt)
        return df

    def remove_identifiers(
        self,
        df,
        joined_dataset_config,
        save_path,
        fmt
    ):
        """Remove identifier columns before modeling dataset creation."""
        identifiers = joined_dataset_config.get("identifiers", [])
        df = df.drop(*identifiers)
        self.write_data(df, save_path, fmt)
        return df

    # -------------------------------------------------------------------------
    # Dataset Info Builders
    # -------------------------------------------------------------------------

    def build_dataset_info(
        self,
        spark,
        fmt,
        tdp_config_list
    ):
        """Load datasets and return list of (DataFrame, alias)."""
        dataset_info = []
        for conf in tdp_config_list:
            path = os.path.join(conf["mount_path"], conf["file"])
            df = self.read_data(spark, path, fmt)
            dataset_info.append((df, conf["name"]))
        return dataset_info

    def generate_output_filename(self, dataset_info):
        """Generate joined dataset filename."""
        return "sandbox_" + "_".join([name for _, name in dataset_info]) + "_linked_anon.csv"

    # -------------------------------------------------------------------------
    # Orchestration
    # -------------------------------------------------------------------------

    def execute(self, config):
        """Run full Spark join pipeline based on config."""
        tdp_config_list = self.load_tdp_list(config)
        joined_dataset_config = self.load_joined_dataset_config(config)

        spark = self.create_spark_session(tdp_config_list)
        spark.sparkContext.setLogLevel("ERROR")

        query = self.validate_and_get_query(joined_dataset_config)

        save_path = joined_dataset_config.get("joined_dataset")
        fmt = save_path.split(".")[-1]

        dataset_info = self.build_dataset_info(spark, fmt, tdp_config_list)

        joined_dataset = self.prepare_joined_dataset(
            spark, dataset_info, joined_dataset_config, query, save_path, fmt
        )

        dataset_names = [d["name"] for d in tdp_config_list]
        print(f"Joined datasets: {dataset_names}")

        self.remove_identifiers(
            joined_dataset,
            joined_dataset_config,
            save_path,
            fmt
        )



class DirectoryJoin(TaskBase):
    
    def join_datasets(self, config):
        output_path = config["joined_dataset"]
        Path(output_path).mkdir(parents=True, exist_ok=True)

        for dataset in config["datasets"]:
            dataset_path = dataset["mount_path"]
            dataset_name = dataset["name"]

            if os.path.isdir(dataset_path):
                for root, dirs, files in os.walk(dataset_path):
                    rel_path = os.path.relpath(root, dataset_path)
                    target_root = os.path.join(output_path, rel_path)
                    os.makedirs(target_root, exist_ok=True)

                    for file in files:
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(target_root, file)

                        if not os.path.exists(dst_file):
                            # Avoid following symlinks
                            if os.path.islink(src_file):
                                continue
                            shutil.copy2(src_file, dst_file, follow_symlinks=False)
                print(f"Merged dataset '{dataset_name}' into '{output_path}'")
            else:
                print(f"Dataset '{dataset_name}' is not a valid directory.")

        print(f"\nAll datasets joined in: {output_path}")


    def execute(self, config):
        # Join the datasets
        self.join_datasets(config)

