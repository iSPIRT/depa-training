
class TaskBase:
    def execute(self, config):
        raise NotImplementedError("Subclasses must implement the execute method.")
