# Bank 1 TDP preprocessor. Derives this bank's mule-detection feature vectors from
# PaySim (its own disjoint account partition) and writes accounts.parquet. All logic
# lives in the shared mule_features module; banks differ only by their index.
from mule_features import run

if __name__ == "__main__":
    run(bank_index=0, n_banks=3)
