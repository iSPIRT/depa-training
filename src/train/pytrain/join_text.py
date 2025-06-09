import pandas as pd
from .task_base import TaskBase

class TextJoin(TaskBase):

    def __init__(self):
        pass

    def load_and_join(self, config):
        merged_data = []
        for ds in config["datasets"]:
            path = ds["mount_path"]+ds["file"]
            df = pd.read_csv(path)[ds["select_variables"]][:ds["num_rows"]]
            df.columns = ["input", "output"]
            merged_data.append(df)

            # debugging - print number of rows
            print(f"Debug | text_join|load_and_join|{ds['file']} count =", df.shape[0])

        # Join the dataframes with standardized column names "input" and "output"
        joined = pd.concat(merged_data, ignore_index=True)
        # debugging - print number of rows
        print(f"Debug | text_join|load_and_join|joined count =", joined.shape[0])

        path = config["joined_dataset"]["output_folder"]+config["joined_dataset"]["output_file"]
        joined.to_csv(path, index=False)
        print(f"Saved joined dataset: {path}")
        
    def execute(self, config):
        self.load_and_join(config)