import os
import random
from typing import *
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm

import scallopy

mnist_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,)
    )
])

class MNISTSumDouble2Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            augmentation: int = 2,
    ):
        # Contains a MNIST dataset
        self.mnist_dataset = torchvision.datasets.MNIST(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.index_map = list(range(len(self.mnist_dataset))) * augmentation
        random.shuffle(self.index_map)

    def __len__(self):
        return int(len(self.mnist_dataset) / 6)

    def __getitem__(self, idx):
        # Get two data points
        (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 6]]
        (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 6 + 1]]
        (c_img, c_digit) = self.mnist_dataset[self.index_map[idx * 6 + 2]]
        (d_img, d_digit) = self.mnist_dataset[self.index_map[idx * 6 + 3]]
        (e_img, e_digit) = self.mnist_dataset[self.index_map[idx * 6 + 4]]
        (f_img, f_digit) = self.mnist_dataset[self.index_map[idx * 6 + 5]]

        # Each data has two images and the GT is the sum of two digits
        return (a_img, b_img, c_img, d_img, e_img, f_img, a_digit * 100 + b_digit * 10 + c_digit + d_digit * 100 + e_digit * 10 + f_digit)

    @staticmethod
    def collate_fn(batch):
        a_imgs = torch.stack([item[0] for item in batch])
        b_imgs = torch.stack([item[1] for item in batch])
        c_imgs = torch.stack([item[2] for item in batch])
        d_imgs = torch.stack([item[3] for item in batch])
        e_imgs = torch.stack([item[4] for item in batch])
        f_imgs = torch.stack([item[5] for item in batch])
        images = torch.stack([a_imgs, b_imgs, c_imgs, d_imgs, e_imgs, f_imgs])
        digits = torch.stack([torch.tensor(item[6]).long() for item in batch])
        return images, digits


def mnist_sum_double_2_loader(data_dir, batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        MNISTSumDouble2Dataset(
            data_dir,
            train=True,
            download=True,
            transform=mnist_img_transform,
        ),
        collate_fn=MNISTSumDouble2Dataset.collate_fn,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=2,
    )

    test_loader = torch.utils.data.DataLoader(
        MNISTSumDouble2Dataset(
            data_dir,
            train=False,
            download=True,
            transform=mnist_img_transform,
        ),
        collate_fn=MNISTSumDouble2Dataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True
    )

    return train_loader, test_loader


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
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
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x


class MNISTSumDouble2Net(nn.Module):
    def __init__(self, provenance, k):
        super(MNISTSumDouble2Net, self).__init__()

        # MNIST Digit Recognition Network
        self.mnist_net = MNISTNet()

        # Scallop Context
        self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
        self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(10)))
        self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
        self.scl_ctx.add_relation("digit_3", int, input_mapping=list(range(10)))
        self.scl_ctx.add_relation("digit_4", int, input_mapping=list(range(10)))
        self.scl_ctx.add_relation("digit_5", int, input_mapping=list(range(10)))
        self.scl_ctx.add_relation("digit_6", int, input_mapping=list(range(10)))
        self.scl_ctx.add_rule("sum(a * 100 + b * 10 + c + d * 100 + e * 10 + f) :- digit_1(a), digit_2(b), digit_3(c), digit_4(d), digit_5(e), digit_6(f)")

        # The `sum_double_2` logical reasoning module
        self.sum_double_2 = self.scl_ctx.forward_function("sum", output_mapping=list(range(200)))

    def forward(self, x: torch.Tensor):
        # First recognize the two digits
        x = self.mnist_net(x.reshape(-1, 1, 28, 28))
        x = x.reshape(6, -1, 10)

        # Then execute the reasoning module; the result is a size 19 tensor
        return self.sum_double_2(digit_1=x[0], digit_2=x[1], digit_3=x[2], digit_4=x[3], digit_5=x[4], digit_6=x[5])


def bce_loss(output, ground_truth):
    (_, dim) = output.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
    return F.nll_loss(output, ground_truth)


class Trainer():
    def __init__(self, train_loader, test_loader, learning_rate, loss, k, provenance):
        self.network = MNISTSumDouble2Net(provenance, k)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.test_loader = test_loader
        if loss == "nll":
            self.loss = nll_loss
        elif loss == "bce":
            self.loss = bce_loss
        else:
            raise Exception(f"Unknown loss function `{loss}`")

    def train_epoch(self, epoch):
        self.network.train()
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        for (data, target) in iter:
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")

    def test_epoch(self, epoch):
        self.network.eval()
        num_items = len(self.test_loader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            iter = tqdm(self.test_loader, total=len(self.test_loader))
            for (data, target) in iter:
                output = self.network(data)
                test_loss += self.loss(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                perc = 100. * correct / num_items
                iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")

    def train(self, n_epochs):
        timings = []
        for epoch in range(1, n_epochs + 1):
            t1 = time.perf_counter()
            self.train_epoch(epoch)
            timings.append(time.perf_counter() - t1)
        print(f"Epoch timings: {np.mean(timings[2:])} \\pm {np.std(timings[2:])}")
        self.test_epoch(epoch)


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("mnist_sum_3")
    parser.add_argument("-e", "--n-epochs", type=int, default=12)
    parser.add_argument("-b", "--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--loss-fn", type=str, default="bce")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--provenance", type=str, default="difftopkproofs")
    parser.add_argument("-k", "--top-k", type=int, default=3)
    args = parser.parse_args()

    # Parameters
    n_epochs = args.n_epochs
    batch_size_train = args.batch_size
    batch_size_test = args.batch_size
    learning_rate = args.learning_rate
    loss_fn = args.loss_fn
    k = args.top_k
    provenance = args.provenance
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Data
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))

    # Dataloaders
    train_loader, test_loader = mnist_sum_double_2_loader(data_dir, batch_size_train, batch_size_test)

    # Create trainer and train
    trainer = Trainer(train_loader, test_loader, learning_rate, loss_fn, k, provenance)
    trainer.train(n_epochs)
