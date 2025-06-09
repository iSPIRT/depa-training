export DATA_DIR=$PWD/../../data
export BRATS_A_INPUT_PATH=$DATA_DIR/brats_A/
export BRATS_A_OUTPUT_PATH=$DATA_DIR/brats_A/preprocessed
export BRATS_B_INPUT_PATH=$DATA_DIR/brats_B/
export BRATS_B_OUTPUT_PATH=$DATA_DIR/brats_B/preprocessed
export BRATS_C_INPUT_PATH=$DATA_DIR/brats_C/
export BRATS_C_OUTPUT_PATH=$DATA_DIR/brats_C/preprocessed
export BRATS_D_INPUT_PATH=$DATA_DIR/brats_D/
export BRATS_D_OUTPUT_PATH=$DATA_DIR/brats_D/preprocessed

docker compose -f docker-compose-preprocess.yml up --remove-orphans
