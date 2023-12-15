export DATA_DIR=$PWD/../../data
export MNIST_1_INPUT_PATH=$DATA_DIR/mnist_1/preprocessed
export MNIST_2_INPUT_PATH=$DATA_DIR/mnist_2/preprocessed
export MNIST_3_INPUT_PATH=$DATA_DIR/mnist_3/preprocessed
export MODEL_INPUT_PATH=$DATA_DIR/modeller/model
export MODEL_OUTPUT_PATH=/tmp/output
mkdir -p $MODEL_OUTPUT_PATH
export CONFIGURATION_PATH=/tmp
cp $PWD/../../config/model_config.json /tmp/model_config.json
cp $PWD/../../config/query_config.json /tmp/query_config.json
docker compose -f docker-compose-train.yml up --remove-orphans
