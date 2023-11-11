export DATA_DIR=$PWD/../../data
export ICMR_INPUT_PATH=$DATA_DIR/icmr/preprocessed
export INDEX_INPUT_PATH=$DATA_DIR/index/preprocessed
export COWIN_INPUT_PATH=$DATA_DIR/cowin/preprocessed
export MODEL_INPUT_PATH=$DATA_DIR/modeller/model
export MODEL_OUTPUT_PATH=/tmp/output
mkdir -p $MODEL_OUTPUT_PATH
export CONFIGURATION_PATH=/tmp
cp $PWD/../../config/model_config.json /tmp/model_config.json
cp $PWD/../../config/query_config.json /tmp/query_config.json
docker compose -f docker-compose-train.yml up --remove-orphans
