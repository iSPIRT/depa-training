# Bank 2 TDP preprocessor. See preprocess_bank_1.py — identical but for its partition.
from mule_features import run

if __name__ == "__main__":
    run(bank_index=1, n_banks=3)
