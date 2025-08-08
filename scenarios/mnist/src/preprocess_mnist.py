import torch
import torchvision
import torchvision.transforms as transforms

mnist_input_folder='/mnt/input/data/'

# Location of preprocessed MNIST dataset
mnist_output_folder='/mnt/output/preprocessed/'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=mnist_input_folder, train=True,
                                        download=True, transform=transform)

# Save the CIFAR10 dataset
torch.save(trainset, mnist_output_folder + 'cifar10-dataset.pth')
