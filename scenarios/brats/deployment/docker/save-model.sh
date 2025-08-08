export MODEL_OUTPUT_PATH=$PWD/../../modeller/models
mkdir -p $MODEL_OUTPUT_PATH
docker compose -f docker-compose-modelsave.yml up --remove-orphans