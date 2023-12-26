import torch
import torch.nn as nn
import torch.optim as optim

# onnx related imports
import onnx
from onnx2pytorch import ConvertModel

from .task_base import TaskBase

class Train(TaskBase):
    def load_data(self, config):
        batch_size = config["batch_size"]

        # Load the dataset from a .pth file
        trainset = torch.load(config["input_dataset_path"])

        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )

    def load_model(self, config):
        onnx_model = onnx.load(config["saved_model_path"])
        model = ConvertModel(onnx_model, experimental=True)
        self.model = model

    def load_optimizer(self, config):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def train(self, config):
        criterion = nn.CrossEntropyLoss()
        for epoch in range(
            config["total_epochs"]
        ):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

    def execute(self, config):
        self.load_data(config)
        self.load_model(config)
        self.load_optimizer(config)
        self.train(config)
