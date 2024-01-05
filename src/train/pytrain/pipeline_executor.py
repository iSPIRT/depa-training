import json
import argparse
from .task_base import TaskBase
from .join import Join
from .private_train import PrivateTrain
from .train import Train

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
                print("Error: Class {step_name} not found or does not inherit from TaskBase.")


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
