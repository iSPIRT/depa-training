# 2023, The DEPA CCR DP Training Reference Implementation
# authors shyam@ispirt.in, sridhar.avs@ispirt.in
#
# Licensed TBD
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Key references / Attributions: https://depa.world/training/contracts
# Key frameworks used : pyspark

import os
import json
import argparse
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import col, column

from .task_base import TaskBase

# Debug Enabled
debug_poc = True

class Join(TaskBase):
    
    def load_tdp_list(self, config, debug=True):
        """
        Extract List of TDP configurations for data joining process from query config
        """
        return config["datasets"]

    def load_joined_dataset_config(cself, config, debug=True):
        """
        Extract List of query join configurations for data joining process from query config.
        """
        return config["joined_dataset"]

    def get_name(self, tdp_config_list, debug=True):
        """
        Extract list of names of all TDP's from query config
        """
        name_list = []
        for c in tdp_config_list:
            name_list.append(c["name"])
        if debug:
            print("Debug |get_name")
        return name_list

    def create_spark_context(self, tdp_config_list, debug=True):
        """
        Create a spark session with app context auto generated from TDP names
        """
        name_list = self.get_name(tdp_config_list)
        context = ""
        for c in name_list:
            context = context + c + "_"
        if debug:
            print("Debug |create_spark_context")
        return SparkSession.builder.appName(context).getOrCreate()

    def generate_query(self, tdp_config_list, joined_dataset_config, debug=True):
        """
        Extract the query logic from the query config. Current implementation extracts query from query config.
        """
        if debug:
            print("Debug |generate_query")
        return joined_dataset_config["joining_query"]

    def dropDupeDfCols(self, df, debug=True):
        """
        Drops Duplicate Columns from the dataframe passed
        """

        newcols = []
        dupcols = []

        for i in range(len(df.columns)):
            if df.columns[i] not in newcols:
                newcols.append(df.columns[i])
            else:
                dupcols.append(i)

        df = df.toDF(*[str(i) for i in range(len(df.columns))])
        for dupcol in dupcols:
            df = df.drop(str(dupcol))
        if debug:
            print("Debug |dropDupeDfCols")
        return df.toDF(*newcols)

    def dp_load_data(self, spark, input_folder, data_file, load=True, debug=True):
        """
        Generic Data Loading Function at Data Provider
        """

        if load:
            input_file = input_folder + data_file
            if debug:
                print("Debug | input_file", input_file)
            data_loaded = spark.read.csv(
                input_file, header=True, inferSchema=True, mode="DROPMALFORMED"
            )
        if debug:
            print("Debug |dp_load_data | input_file", data_loaded.count())
            data_loaded.show(2)
        return data_loaded

    def create_view(self, dataset_name, view_name):
        """
        Create the temp query-able views for spark query processing
        """
        return dataset_name.createOrReplaceTempView(view_name)

    def ccr_prepare_joined_dataset(self, dataset_info, query, model_file, debug=True):
        """
        In CCR/Sandbox we are creating the joined dataset anon from the three TDP and joined data configurations . This function abstracts this and returns the dataet.
        """
        # Create the temp query-able views
        for c in dataset_info:
            self.create_view(c[0], c[1])
        # Query Execution
        retun_dataset = spark.sql(query).cache()
        retun_dataset = retun_dataset.drop("_c0")
        # Save the dataset
        retun_dataset.toPandas().to_csv(model_file)
        if debug:
            print("Debug | ccr_prepare_joined_dataset | Dataset Created ", model_file)
            retun_dataset.show(2)
        return retun_dataset

    def ccr_prepare_joined_dataset_full(
        self, spark, dataset_info, joined_dataset_config, query, model_file, debug=True
    ):
        """
        In CCR/Sandbox we are creating the joined dataset anon from the three TDPs. This function abstracts this and returns the dataet.
        """
        # Create the temp query-able views
        for c in dataset_info:
            self.create_view(c[0], c[1])

        # Query Execution
        return_dataset = spark.sql(query).cache()
        return_dataset = return_dataset.drop("_c0")
        return_dataset_step1 = return_dataset.dropDuplicates()
        return_dataset_step2 = self.dropDupeDfCols(return_dataset_step1)
        drop_columns = joined_dataset_config["drop_columns"]
        return_dataset_step3 = return_dataset_step2.drop(*drop_columns)
        return_dataset_final = return_dataset_step3

        return_dataset_final.toPandas().to_csv(model_file)

        if debug:
            print(
                "Debug | ccr_prepare_joined_dataset_full|joint_dataset| count =",
                return_dataset.count(),
            )
            print(
                "Debug | ccr_prepare_joined_dataset_full|joint_dataset|step1 count =",
                return_dataset_step1.count(),
            )
            print(
                "Debug | ccr_prepare_joined_dataset_full|joint_dataset|step2 count =",
                return_dataset_step2.count(),
            )
            print(
                "Debug | ccr_prepare_joined_dataset_full|joint_dataset|step3 count =",
                return_dataset_step3.count(),
            )
            print(
                "Debug | ccr_prepare_joined_dataset_full|joint_dataset|final count =",
                return_dataset_final.count(),
            )
            return_dataset.show(2)
            print(
                "Debug | ccr_prepare_joined_dataset_full |Dataset Created ", model_file
            )

        return return_dataset_final

    def ccr_create_joined_dataset_wo_identifiers(
        self, joined_dataset, joined_dataset_config, modelfile, debug=True
    ):
        identifiers = joined_dataset_config["identifiers"]
        return_dataset = joined_dataset.drop(*identifiers)
        # Modeling dataset
        return_dataset.toPandas().to_csv(modelfile)

        if debug_poc:
            print(
                "Debug | ccr_create_joined_dataset_wo_identifiers|joint_dataset| count  =",
                joined_dataset.count(),
            )
            print(
                "Debug | ccr_create_joined_dataset_wo_identifiers|return_dataset| count =",
                return_dataset.count(),
            )
            return_dataset.show(2)

        return return_dataset

    def generate_data_info(self, spark, tdp_config_list):
        """
        Extracts the list of loaded data set along with its alias from TDP config.
        Required for Spark view creation-create_view
        """
        lis = []
        for c in tdp_config_list:
            l = []
            l.append(self.dp_load_data(spark, c["mount_path"], c["file"]))
            l.append(c["name"])
            lis.append(l)
        return lis

    def generate_base_query_dataset(self, dataset_info):
        """
        This will generate the base joined dataset for ccr_prepare_joined_dataset_full function.
        """
        # sandbox_icmr_cowin_index_linked_anon
        file_str = "sandbox_"
        for c in dataset_info:
            file_str = file_str + c[1] + "_"
        file_str = file_str + "linked_anon.csv"

        if debug_poc:
            print("Debug | generate_base_query_dataset | file generated ", file_str)

        return file_str

    def execute(self, config):
        """
        Final Execution Function
        """
        tdp_config_list = self.load_tdp_list(config)
        joined_dataset_config = self.load_joined_dataset_config(config)
        spark = self.create_spark_context(
            tdp_config_list
        )  # currently treated as a global instance but can be converted into a specific instance for multiple pipelines
        query = self.generate_query(tdp_config_list, joined_dataset_config)
        dataset_info = self.generate_data_info(spark, tdp_config_list)
        model_output_folder = joined_dataset_config["model_output_folder"]
        # sandbox_joined_anon_simplified=ccr_prepare_joined_dataset_full(dataset_info,query,joined_dataset_config["model_file"],debug=True)
        model_file = joined_dataset_config["joined_dataset"]
        sandbox_joined_anon_simplified = self.ccr_prepare_joined_dataset_full(
            spark, dataset_info, joined_dataset_config, query, model_file, debug=True
        )
        print("Generating aggregated data in " + model_output_folder + model_file)
        sandbox_joined_without_key_identifiers = (
            self.ccr_create_joined_dataset_wo_identifiers(
                sandbox_joined_anon_simplified,
                joined_dataset_config,
                model_output_folder + model_file,
                True,
            )
        )

