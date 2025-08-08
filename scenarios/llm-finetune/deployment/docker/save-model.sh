export MODEL_OUTPUT_PATH=$PWD/../../model
export CONFIG_PATH=$PWD/../../config/model_repo_config.json
mkdir -p $MODEL_OUTPUT_PATH
docker compose -f docker-compose-modelsave.yml up --remove-orphans