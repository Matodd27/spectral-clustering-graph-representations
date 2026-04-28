from .CIFAR10.CIFAR10_simclr import SimCLR as SimCLR_CIFAR10
from .MNIST.mnist_simclr import SimCLR as SimCLR_MNIST
from .FashionMNIST.FashionMNIST_simclr import SimCLR as SimCLR_FashionMNIST

__all__ = ['SimCLR_CIFAR10', 'SimCLR_MNIST', 'SimCLR_FashionMNIST']