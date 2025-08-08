# 2025, DEPA Foundation
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

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import sha2, concat_ws # Hashing Related functions
from pyspark.sql.functions import col , column

##**Key Configuration Variables**"""

# Debug Enabled
debug_poc=True

# Model Input and output folders
model_input_folder='/mnt/input/data/'

# POC State-wise dummy data
index_file ='poc_data_statewarroom_data.csv'

# Used for loading/Process at DP (Ideally done at DP, we just pick up anonymised/tokenised datasets)
load_process_dp_data=True

# Data Provider Level - Preprocess Locations
dp_index_output_folder='/mnt/output/preprocessed/'

# DP Standardisation Non Anon Files
dp_index_std_nonanon_file ='dp_index_standardised_nonanon.csv'

# DP Standardisation  Anon | Tokenised Files
dp_index_std_anon_file ='dp_index_standardised_anon.csv'

dp_joined_dataset_identifiers_file='sandbox_icmr_cowin_index_linked_anon.csv'
dp_joined_dataset_wo_identifiers_file='sandbox_icmr_cowin_index_without_key_identifiers.csv'

# Query Execution
ccr_joining_query = "select * from ICMR_A a, INDEX_A b, COWIN_A c " + "where b.pk_mobno_hashed == a.pk_mobno_hashed and b.pk_mobno_hashed == c.pk_mobno_hashed"

"""# Setting up spark session"""

spark = SparkSession.builder.appName('CCR_DEPA_COVID_POC_Code').getOrCreate()

"""# Common Utility Functions

"""

def dropDupeDfCols(df):
  '''
  Duplicate Columns Drop 
  Reference
  # https://stackoverflow.com/questions/46944493/removing-duplicate-columns-after-a-df-join-in-spark
  '''

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

  return df.toDF(*newcols)


def dp_load_data(input_folder,data_file,load=True,debug=True):
  """
  Generic Data Loading Function at Data Provider
  """

  if load:
    input_file=input_folder+data_file
    if debug: 
      print("Debug | input_file",input_file)
    data_loaded= spark.read.csv(
    input_file, 
    header=True, 
    inferSchema=True,
    mode="DROPMALFORMED"
  )
  if debug: 
      print("Debug | input_file",data_loaded.count())
      data_loaded.show()
  return data_loaded


def dp_process_index_full(input_folder,data_file,load=True,debug=True):
    """
    Standardisation of Index data done at Data Provider Infrastructure
    Suggested to have some standardisations defined by the DEP/ Data Consumer (DC) for easier joining process
    """

    if load:
      input_file=input_folder+data_file
    
    if debug: 
      print("Debug | input_file",input_file)
    
    data_loaded= spark.read.csv(
      input_file, 
      header=True, 
      inferSchema=True,
      mode="DROPMALFORMED"
      )
    
    if debug: 
      print("Debug | input_file",data_loaded.count())
      data_loaded.show()

    sandbox_dp_index=data_loaded
    if debug:
      sandbox_dp_index.show()

    # Standardisations
    sandbox_dp_index = sandbox_dp_index.withColumnRenamed('icmrnumber', 'pk_icmrno') 
    sandbox_dp_index = sandbox_dp_index.withColumnRenamed('pcnumber', 'pk_mobno') 
    sandbox_dp_index = sandbox_dp_index.withColumnRenamed('srfnumber', 'ref_srfno') 
    sandbox_dp_index = sandbox_dp_index.withColumnRenamed('labcode', 'ref_labid') 
   

    # Standardisations
    # Exclusion Variables for prefix
    do_not_change=['pk_icmrno','pk_mobno', 'ref_srfno', 'ref_labid']

    # Prefix the source for all other columns
    for i in sandbox_dp_index.columns:
      if i not in do_not_change:
        sandbox_dp_index = sandbox_dp_index.withColumnRenamed(i,'index_'+i)

    # Create the Output
    sandbox_dp_index.toPandas().to_csv(dp_index_output_folder+ dp_index_std_nonanon_file)

    # Anonymisation of key identifiers
    sandbox_dp_index_anon = sandbox_dp_index.withColumn('pk_icmrno_hashed', sha2(concat_ws("", sandbox_dp_index.pk_icmrno),256)) \
    .withColumn("ref_srfno_hashed", sha2(concat_ws("", sandbox_dp_index.ref_srfno),256)) \
    .withColumn("pk_mobno_hashed", sha2(concat_ws("", sandbox_dp_index.pk_mobno),256)) \
    .withColumn("ref_labid_hashed", sha2(concat_ws("", sandbox_dp_index.ref_labid),256)) \
    .drop("pk_mobno").drop("ref_srfno").drop("pk_icmrno").drop("ref_labid").cache()

    sandbox_dp_index_anon.toPandas().to_csv(dp_index_output_folder + dp_index_std_anon_file)
    
    if debug_poc:
      print("Debug | Dataset Created ", "sandbox_index_processed_anon")
      print("Debug | sandbox_index_nonanon count =", sandbox_dp_index.count())
      print("Debug | sandbox_index_anon count =", sandbox_dp_index_anon.count())
      sandbox_dp_index.show()
      sandbox_dp_index_anon.show()

    return sandbox_dp_index_anon

#State War Room
#index_mockdata_testing=dp_load_data(model_input_folder,index_file,True,True)
#ccr_sandbox_index=dp_process_index(index_mockdata_testing,True)
ccr_sandbox_index=dp_process_index_full(model_input_folder,index_file,True,True)

