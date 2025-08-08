export DATA_DIR=$PWD/../../data
export MODEL_DIR=$PWD/../../modeller

export BRATS_A_INPUT_PATH=$DATA_DIR/brats_A/preprocessed
export BRATS_B_INPUT_PATH=$DATA_DIR/brats_B/preprocessed
export BRATS_C_INPUT_PATH=$DATA_DIR/brats_C/preprocessed
export BRATS_D_INPUT_PATH=$DATA_DIR/brats_D/preprocessed

export MODEL_INPUT_PATH=$MODEL_DIR/models
cp $MODEL_DIR/../src/load_base_model.py $MODEL_DIR/models/load_base_model.py
cp $MODEL_DIR/../src/custom_dataset.py $MODEL_DIR/models/custom_dataset.py

# export MODEL_OUTPUT_PATH=/tmp/output
export MODEL_OUTPUT_PATH=$MODEL_DIR/output
mkdir -p $MODEL_OUTPUT_PATH 

# export CONFIGURATION_PATH=/tmp
export CONFIGURATION_PATH=$PWD/../../config
# cp $PWD/../../config/pipeline_config.json /tmp/pipeline_config.json

docker compose -f docker-compose-train.yml up --remove-orphans
