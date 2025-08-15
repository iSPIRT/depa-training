export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export MODEL_DIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

export MNIST_INPUT_PATH=$DATA_DIR/preprocessed

export MODEL_INPUT_PATH=$MODEL_DIR/models
cp $MODEL_DIR/../src/class_definitions.py $MODEL_DIR/models/class_definitions.py

# export MODEL_OUTPUT_PATH=/tmp/output
export MODEL_OUTPUT_PATH=$MODEL_DIR/output
rm -rf $MODEL_OUTPUT_PATH
mkdir -p $MODEL_OUTPUT_PATH 

export CONFIGURATION_PATH=$REPO_ROOT/scenarios/$SCENARIO/config
# export CONFIGURATION_PATH=/tmp
# cp $PWD/../../config/pipeline_config.json /tmp/pipeline_config.json

docker compose -f docker-compose-train.yml up --remove-orphans
