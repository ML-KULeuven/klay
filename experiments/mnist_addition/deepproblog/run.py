import argparse
from json import dumps

import torch
import torch.nn as nn

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
        self.activation = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return self.activation(x)


def main(nb_digits: int, nb_epochs: int, batch_size: int, learning_rate: float):
    train_set = addition(nb_digits, "train")
    test_set = addition(nb_digits, "test")

    network = LeNet()
    net = Network(network, "mnist_net", batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    model = Model("experiments/mnist_addition/deepproblog/addition.pl", [net])
    model.set_engine(ExactEngine(model), cache=True)

    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("test", MNIST_test)

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train = train_model(model, loader, nb_epochs, log_iter=10, profile=0)
    train.logger.comment(dumps(model.get_hyperparameters()))
    train.logger.comment(
        "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nb_digits', type=int, default=2)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--nb_epochs', type=int, default=12)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0003)
    args = parser.parse_args()

    main(args.nb_digits, args.nb_epochs, args.batch_size, args.learning_rate)