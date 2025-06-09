export MODEL_OUTPUT_PATH=$PWD/../../model
mkdir -p $MODEL_OUTPUT_PATH
docker compose -f docker-compose-modelsave.yml up --remove-orphans
