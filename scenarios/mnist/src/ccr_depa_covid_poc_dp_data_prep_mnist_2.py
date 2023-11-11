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
#Crirical Library Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import sha2, concat_ws # Hashing Related functions
from pyspark.sql.functions import col , column

##**Key Configuration Variables**"""

# Debug Enabled
debug_poc=True

# Model Input and output folders
model_input_folder='/mnt/depa_ccr_poc/data/'

# POC State-wise dummy data
mnist_2_file ='poc_mnist_2_data.csv'

# Used for loading/Process at DP (Ideally done at DP, we just pick up anonymised/tokenised datasets)
load_process_dp_data=True

# Data Provider Level - Preprocess Locations
dp_mnist_2_output_folder='/mnt/output/mnist_2/'

# DP Standardisation Non Anon Files
dp_mnist_2_std_nonanon_file ='dp_mnist_2_standardised_nonanon.csv'

# DP Standardisation  Anon | Tokenised Files
dp_mnist_2_std_anon_file ='dp_mnist_2_standardised_anon.csv'

# In the CCR
model_output_folder='/mnt/depa_ccr_poc/output/'

dp_joined_dataset_identifiers_file='sandbox_mnist_linked_anon.csv'
dp_joined_dataset_wo_identifiers_file='sandbox_mnist_without_key_identifiers.csv'



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


def dp_process_mnist_2_full(input_folder,data_file,load=True,debug=True):
    """
    Standardisation of mnist_2 data done at Data Provider Infrastructure
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

    sandbox_dp_mnist_2=data_loaded
    if debug:
      sandbox_dp_mnist_2.show()


   


    # Create the Output
    sandbox_dp_mnist_2.toPandas().to_csv(dp_mnist_2_output_folder+ dp_mnist_2_std_nonanon_file)


    sandbox_dp_mnist_2.toPandas().to_csv(dp_mnist_2_output_folder + dp_mnist_2_std_anon_file)
    
    if debug_poc:
      print("Debug | Dataset Created ", "sandbox_mnist_2_processed_anon")
      print("Debug | sandbox_mnist_2_nonanon count =", sandbox_dp_mnist_2.count())
      print("Debug | sandbox_mnist_2_anon count =", sandbox_dp_mnist_2.count())
      sandbox_dp_mnist_2.show()
      sandbox_dp_mnist_2.show()

    return sandbox_dp_mnist_2


ccr_sandbox_mnist_2=dp_process_mnist_2_full(model_input_folder,mnist_2_file,True,True)

