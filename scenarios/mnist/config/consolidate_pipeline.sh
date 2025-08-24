#! /bin/bash

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCENARIO=mnist

template_path="$REPO_ROOT/scenarios/$SCENARIO/config/templates"
model_config_path="$REPO_ROOT/scenarios/$SCENARIO/config/model_config.json"
data_config_path="$REPO_ROOT/scenarios/$SCENARIO/config/dataset_config.json"
loss_config_path="$REPO_ROOT/scenarios/$SCENARIO/config/loss_config.json"
train_config_path="$REPO_ROOT/scenarios/$SCENARIO/config/train_config.json"
eval_config_path="$REPO_ROOT/scenarios/$SCENARIO/config/eval_config.json"
join_config_path="$REPO_ROOT/scenarios/$SCENARIO/config/join_config.json"
pipeline_config_path="$REPO_ROOT/scenarios/$SCENARIO/config/pipeline_config.json"

# populate "model_config", "data_config", and "loss_config" keys in train config
train_config=$(cat $template_path/train_config_template.json)

# Only merge if the file exists
if [[ -f "$model_config_path" ]]; then
    model_config=$(cat $model_config_path)
    train_config=$(echo "$train_config" | jq --argjson model "$model_config" '.config.model_config = $model')
fi

if [[ -f "$data_config_path" ]]; then
    data_config=$(cat $data_config_path)
    train_config=$(echo "$train_config" | jq --argjson data "$data_config" '.config.dataset_config = $data')
fi

if [[ -f "$loss_config_path" ]]; then
    loss_config=$(cat $loss_config_path)
    train_config=$(echo "$train_config" | jq --argjson loss "$loss_config" '.config.loss_config = $loss')
fi

if [[ -f "$eval_config_path" ]]; then
    eval_config=$(cat $eval_config_path)
    # Get all keys from eval_config and copy them to train_config
    for key in $(echo "$eval_config" | jq -r 'keys[]'); do
        train_config=$(echo "$train_config" | jq --argjson eval "$eval_config" --arg key "$key" '.config[$key] = $eval[$key]')
    done
fi

# save train_config
echo "$train_config" > $train_config_path

# prepare pipeline config from join_config.json (first dict "config") and train_config.json (second dict "config")
pipeline_config=$(cat $template_path/pipeline_config_template.json)

# Only merge join_config if the file exists
if [[ -f "$join_config_path" ]]; then
    join_config=$(cat $join_config_path)
    pipeline_config=$(echo "$pipeline_config" | jq --argjson join "$join_config" '.pipeline += [$join]')
fi

# Always merge train_config as it's required
pipeline_config=$(echo "$pipeline_config" | jq --argjson train "$train_config" '.pipeline += [$train]')

# save pipeline_config to pipeline_config.json
echo "$pipeline_config" > $pipeline_config_path