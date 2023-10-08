#!/bin/bash

# Wait for query configuration to be available 
until [ "$(ls -A "/mnt/remote/config/query_config.json")" != "" ]
do   
  sleep 10
done
echo "query configuration is available"

# Wait for all mount points in the input dataset to be available
for path in `jq -r .datasets[].mount_path /mnt/remote/config/query_config.json`; do
  echo "waiting for path $path..."
  until [ "$(ls -A $path)" != "" ]
  do   
    sleep 10
  done
  echo "$path is available"
done

echo "Joining datasets with configuration:"
cat /mnt/remote/config/query_config.json
python3.9 ccr_join.py --query-config /mnt/remote/config/query_config.json

# Wait for model configuration to be available 
until [ "$(ls -A "/mnt/remote/config/model_config.json")" != "" ]
do   
  sleep 10
done
echo "model configuration is available"

echo "Training model with configuration:"
cat /mnt/remote/config/model_config.json

python3.9 ccr_train.py --model-config /mnt/remote/config/model_config.json
