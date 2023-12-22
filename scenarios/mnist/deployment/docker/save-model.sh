export MODEL_OUTPUT_PATH=$PWD/../../data/model
mkdir -p $MODEL_OUTPUT_PATH
docker compose -f docker-compose-save-model.yml up --remove-orphans
