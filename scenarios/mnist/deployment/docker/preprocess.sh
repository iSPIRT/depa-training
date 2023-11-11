export DATA_DIR=$PWD/../../data
export MNIST_1_INPUT_PATH=$DATA_DIR/mnist_1
export MNIST_1_OUTPUT_PATH=$DATA_DIR/mnist_1/preprocessed
export MNIST_2_INPUT_PATH=$DATA_DIR/mnist_2
export MNIST_2_OUTPUT_PATH=$DATA_DIR/mnist_2/preprocessed
export MNIST_3_INPUT_PATH=$DATA_DIR/mnist_3
export MNIST_3_OUTPUT_PATH=$DATA_DIR/mnist_3/preprocessed
docker compose -f docker-compose-preprocess.yml up --remove-orphans
