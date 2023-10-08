export MODEL_OUTPUT_PATH=$PWD/../../data/modeller/model
mkdir -p $MODEL_OUTPUT_PATH
docker compose -f docker-compose-modelsave.yml up --remove-orphans
