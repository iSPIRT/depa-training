export MODEL_OUTPUT_PATH=$REPO_ROOT/scenarios/$SCENARIO/modeller/models
mkdir -p $MODEL_OUTPUT_PATH
docker compose -f docker-compose-modelsave.yml up --remove-orphans