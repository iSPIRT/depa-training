export DATA_DIR=$PWD/../../data
export MEDQA_INPUT_PATH=$DATA_DIR/medqa  
export MEDQA_OUTPUT_PATH=$DATA_DIR/medqa/preprocessed
export MEDQUAD_INPUT_PATH=$DATA_DIR/medquad
export MEDQUAD_OUTPUT_PATH=$DATA_DIR/medquad/preprocessed
export CHATDOCTOR_INPUT_PATH=$DATA_DIR/chatdoctor
export CHATDOCTOR_OUTPUT_PATH=$DATA_DIR/chatdoctor/preprocessed
export CONFIG_PATH=$PWD/../../config
docker compose -f docker-compose-preprocess.yml up --remove-orphans