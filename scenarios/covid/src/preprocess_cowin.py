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

#Crirical Library Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import sha2, concat_ws # Hashing Related functions
from pyspark.sql.functions import col , column


"""##**Key Configuration Variables**"""

# Debug Enabled
debug_poc=False

# Model Input and output folders
model_input_folder='/mnt/input/data/'

cowin_file='poc_data_cowin_data.csv'

# Used for loading/Process at DP (Ideally done at DP, we just pick up anonymised/tokenised datasets)
load_process_dp_data=True

# Data Provider Level - Preprocess Locations
dp_cowin_output_folder='/mnt/output/preprocessed/'

# DP Standardisation Non Anon Files
dp_cowin_std_nonanon_file ='dp_cowin_standardised_nonanon.csv'

# DP Standardisation  Anon | Tokenised Files
dp_cowin_std_anon_file ='dp_cowin_standardised_anon.csv'


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

def dp_process_cowin_full(input_folder,data_file,load=True,debug=True):
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


    # Standardisations
    sandbox_dp_cowin=data_loaded
    sandbox_dp_cowin = sandbox_dp_cowin.withColumnRenamed('beneficiary_reference_id', 'pk_beneficiary_id')
    sandbox_dp_cowin = sandbox_dp_cowin.withColumnRenamed('cowin_registered_mobile', 'pk_mobno')
    sandbox_dp_cowin = sandbox_dp_cowin.withColumnRenamed('uhid', 'ref_uhid')
    sandbox_dp_cowin = sandbox_dp_cowin.withColumnRenamed('id_verified', 'ref_id_verified')

    # Exclusion Variables for prefix
    do_not_change=['pk_beneficiary_id','pk_mobno', 'ref_uhid','ref_id_verified']

    for i in sandbox_dp_cowin.columns:
      if i not in do_not_change:
        sandbox_dp_cowin = sandbox_dp_cowin.withColumnRenamed(i,'cowin_'+i)
    
    sandbox_dp_cowin.toPandas().to_csv(dp_cowin_output_folder+dp_cowin_std_nonanon_file)
  
    # Anonymisation of key identifiers
    sandbox_dp_cowin_anon = sandbox_dp_cowin.withColumn('pk_beneficiary_id_hashed', sha2(concat_ws("", sandbox_dp_cowin.pk_beneficiary_id),256)) \
    .withColumn("ref_uhid_hashed", sha2(concat_ws("", sandbox_dp_cowin.ref_uhid),256)) \
    .withColumn("pk_mobno_hashed", sha2(concat_ws("", sandbox_dp_cowin.pk_mobno),256)) \
    .withColumn("ref_id_verified_hashed", sha2(concat_ws("", sandbox_dp_cowin.ref_id_verified),256)) \
    .drop("pk_mobno").drop("pk_beneficiary_id").drop("ref_uhid").drop("ref_id_verified").cache()

    sandbox_dp_cowin_anon.toPandas().to_csv(dp_cowin_output_folder + dp_cowin_std_anon_file)
    
    if debug_poc:
      print("Debug | Dataset Created ", "sandbox_cowin_processed_anon")
      print("Debug | sandbox_cowin_nonanon count =", sandbox_dp_cowin.count())
      print("Debug | sandbox_cowin_anon count =", sandbox_dp_cowin_anon.count())
      sandbox_dp_cowin.show()
      sandbox_dp_cowin_anon.show()

    return sandbox_dp_cowin_anon

ccr_sandbox_cowin=dp_process_cowin_full(model_input_folder, cowin_file,True,False)



