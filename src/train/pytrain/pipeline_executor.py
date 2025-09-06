# 2025 DEPA Foundation
#
# This work is dedicated to the public domain under the CC0 1.0 Universal license.
# To the extent possible under law, DEPA Foundation has waived all copyright and 
# related or neighboring rights to this work. 
# CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/)
#
# This software is provided "as is", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a 
# particular purpose and noninfringement. In no event shall the authors or copyright
# holders be liable for any claim, damages or other liability, whether in an action
# of contract, tort or otherwise, arising from, out of or in connection with the
# software or the use or other dealings in the software.
#
# For more information about this framework, please visit:
# https://depa.world/training/depa_training_framework/

import json
import argparse
from .task_base import TaskBase
from .join import *
from .dl_train import Train_DL
from .xgb_train import Train_XGB

class PipelineExecutor:
    def __init__(self):
        self.steps = []

    def load_pipeline_from_json(self, json_file):
        """Load pipeline steps from a JSON file."""
        with open(json_file, 'r') as file:
            pipeline_config = json.load(file)
            self.steps = pipeline_config.get('pipeline', [])

    def execute_pipeline(self):
        """Execute the pipeline."""
        for step in self.steps:
            step_name = step.get('name')
            step_config = step.get('config', {})
            
            # Dynamically instantiate the class based on the step name
            step_class = globals().get(step_name)
            if step_class is not None and issubclass(step_class, TaskBase):
                # Instantiate the class and execute the step with configuration
                step_instance = step_class()
                step_instance.execute(step_config)
            else:
                print(f"Error: Class {step_name} not found or does not inherit from TaskBase.")


def main():
    parser = argparse.ArgumentParser(description='Execute a pipeline from a JSON configuration file.')
    parser.add_argument('config_file', type=str, help='Path to the pipeline configuration file (JSON)')
    args = parser.parse_args()

    # Create a PipelineExecutor, load the pipeline from JSON, and execute it
    pipeline_executor = PipelineExecutor()
    pipeline_executor.load_pipeline_from_json(args.config_file)
    pipeline_executor.execute_pipeline()


if __name__ == "__main__":
    main()
