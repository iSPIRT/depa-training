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

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import sha2, concat_ws # Hashing Related functions
from pyspark.sql.functions import col , column

##**Key Configuration Variables**"""

# Debug Enabled
debug_poc=False

# Model Input folder
icmr_input_folder='/mnt/input/data/'

# POC State-wise dummy data
icmr_file='poc_data_icmr_data.csv'

# Data Provider Level - Preprocess Locations
dp_icmr_output_folder='/mnt/output/preprocessed/'

# DP Standardisation Non Anon Files
dp_icmr_std_nonanon_file ='dp_icmr_standardised_nonanon.csv'

# DP Standardisation  Anon | Tokenised Files
dp_icmr_std_anon_file ='dp_icmr_standardised_anon.csv'

"""# Setting up spark session"""

spark = SparkSession.builder.appName('CCR_DEPA_COVID_POC_Code').getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

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



def dp_process_icmr_full(input_folder,data_file,load=True,debug=True):
    """
    Standardisation of ICMR data done at Data Provider Infrastructure
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
    sandbox_dp_icmr=data_loaded
    if debug:
      sandbox_dp_icmr.show()

    sandbox_dp_icmr = sandbox_dp_icmr.withColumnRenamed('icmrnumber', 'pk_icmrno')
    sandbox_dp_icmr = sandbox_dp_icmr.withColumnRenamed('icmr_patient_mobilenumber', 'pk_mobno')
    sandbox_dp_icmr = sandbox_dp_icmr.withColumnRenamed('srfnumber', 'ref_srfno')
    sandbox_dp_icmr = sandbox_dp_icmr.withColumnRenamed('a_lab_id', 'fk_icmr_labid') 
    sandbox_dp_icmr = sandbox_dp_icmr.withColumnRenamed('a_sample_genetic_strain', 'fk_genetic_strain') 
    sandbox_dp_icmr.cache()

    do_not_change=['pk_icmrno','pk_mobno', 'ref_srfno','fk_icmr_labid','fk_genetic_strain']

    # Standardisations
    for i in sandbox_dp_icmr.columns:
      if i not in do_not_change:
        sandbox_dp_icmr = sandbox_dp_icmr.withColumnRenamed(i,'icmr_'+i)

    # Create the Output
    sandbox_dp_icmr.toPandas().to_csv(dp_icmr_output_folder+ dp_icmr_std_nonanon_file)

    # Anonymisation of key identifiers
    sandbox_dp_icmr_anon = sandbox_dp_icmr.withColumn('pk_mobno_hashed', sha2(concat_ws("", sandbox_dp_icmr.pk_mobno),256)) \
    .withColumn("ref_srfno_hashed", sha2(concat_ws("", sandbox_dp_icmr.ref_srfno),256)) \
    .withColumn("pk_icmrno_hashed", sha2(concat_ws("", sandbox_dp_icmr.pk_icmrno),256)) \
    .withColumn('fk_icmr_labid_hashed', sha2(concat_ws("", sandbox_dp_icmr.fk_icmr_labid),256)) \
    .drop("pk_mobno").drop("ref_srfno").drop("pk_icmrno").drop("fk_icmr_labid").cache()

    sandbox_dp_icmr_anon.toPandas().to_csv(dp_icmr_output_folder + dp_icmr_std_anon_file)
    if debug_poc:
      print("Debug | sandbox_icmr_anon count =", sandbox_dp_icmr_anon.count())
      print("Debug | Dataset Created ", "sandbox_icmr_processed_nonanon")
      #icmrdata.show()
      sandbox_dp_icmr.show()
      sandbox_dp_icmr_anon.show()
    
    return sandbox_dp_icmr_anon

#icmr_mockdata_testing=dp_load_data(model_input_folder,icmr_file,True,False)

ccr_sandbox_icmr=dp_process_icmr_full(icmr_input_folder,icmr_file,True,False)


