export DATA_DIR=$PWD/../../data
export MNIST_INPUT_PATH=$DATA_DIR/preprocessed
export MODEL_INPUT_PATH=$DATA_DIR/model
export MODEL_OUTPUT_PATH=/tmp/output
mkdir -p $MODEL_OUTPUT_PATH
export CONFIGURATION_PATH=/tmp
cp $PWD/../../config/pipeline_config.json /tmp/pipeline_config.json
docker compose -f docker-compose-train.yml up --remove-orphans
