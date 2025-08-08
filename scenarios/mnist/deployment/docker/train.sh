export DATA_DIR=$PWD/../../data
export MODEL_DIR=$PWD/../../modeller
export MNIST_INPUT_PATH=$DATA_DIR/preprocessed
export MODEL_INPUT_PATH=$MODEL_DIR/models
export MODEL_OUTPUT_PATH=$MODEL_DIR/output
mkdir -p $MODEL_OUTPUT_PATH
export CONFIGURATION_PATH=$PWD/../../config
# export CONFIGURATION_PATH=/tmp
# cp $PWD/../../config/pipeline_config.json /tmp/pipeline_config.json
docker compose -f docker-compose-train.yml up --remove-orphans
