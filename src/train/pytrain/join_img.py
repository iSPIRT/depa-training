import os
import shutil

from .task_base import TaskBase

class ImageJoin(TaskBase):
    
    def join_datasets(self, config):
        output_path = config["joined_dataset"]
        os.makedirs(output_path, exist_ok=True)

        for dataset in config["datasets"]:
            dataset_path = dataset["mount_path"]
            dataset_name = dataset["name"]

            if os.path.isdir(dataset_path):
                for root, dirs, files in os.walk(dataset_path):
                    rel_path = os.path.relpath(root, dataset_path)
                    target_root = os.path.join(output_path, rel_path)
                    os.makedirs(target_root, exist_ok=True)

                    for file in files:
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(target_root, file)

                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
                print(f"Merged dataset '{dataset_name}' into '{output_path}'")
            else:
                print(f"Dataset '{dataset_name}' is not a valid directory.")

        print(f"\nAll datasets joined in: {output_path}")


    def execute(self, config):
        # Join the datasets
        self.join_datasets(config)

