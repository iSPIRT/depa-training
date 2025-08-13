export DATA_DIR=$PWD/../../data
export MODEL_DIR=$PWD/../../modeller

export ICMR_INPUT_PATH=$DATA_DIR/icmr/preprocessed
export INDEX_INPUT_PATH=$DATA_DIR/index/preprocessed
export COWIN_INPUT_PATH=$DATA_DIR/cowin/preprocessed

export MODEL_INPUT_PATH=$MODEL_DIR/models
cp $MODEL_DIR/../src/class_definitions.py $MODEL_DIR/models/class_definitions.py

# export MODEL_OUTPUT_PATH=/tmp/output
export MODEL_OUTPUT_PATH=$MODEL_DIR/output
rm -rf $MODEL_OUTPUT_PATH
mkdir -p $MODEL_OUTPUT_PATH

export CONFIGURATION_PATH=$PWD/../../config
# export CONFIGURATION_PATH=/tmp
# cp $PWD/../../config/pipeline_config.json /tmp/pipeline_config.json

docker compose -f docker-compose-train.yml up --remove-orphans
