import argparse
import io
import zipfile
from pathlib import Path
from time import perf_counter

import requests

import klay
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def download_visudo_dataset(grid_size: int):
    data_path = Path(__file__).parent / Path("tmp")
    if data_path.exists():
        return

    print("-> Downloading Visual Sudoku Dataset...")
    r = requests.get(f"https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/"
                     f"ViSudo-PC_dimension::{grid_size}_datasets::mnist_strategy::simple.zip")
    print("-> Extracting...")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(Path(__file__).parent)


class SudokuDataset(Dataset):
    def __init__(self, partition: str, grid_size: int = 4, transform=None):
        super().__init__()
        data_path = Path(__file__).parent / (f"tmp/ViSudo-PC/ViSudo-PC_dimension::4_datasets::"
                                             f"mnist_strategy::simple/dimension::{grid_size}/datasets:"
                                             f":mnist/strategy::simple/strategy::simple/numTrain::00100/"
                                             f"numTest::00100/numValid::00100/corruptChance::0.50/"
                                             f"overlap::0.00/split::11")
        features_file = Path(data_path) / f'{partition}_puzzle_pixels.txt'
        labels_file = Path(data_path) / f'{partition}_puzzle_labels.txt'
        labels = np.loadtxt(labels_file, delimiter="\t", dtype=bool)
        features = np.loadtxt(features_file, delimiter="\t", dtype=np.float32)
        self.images = torch.as_tensor(features)
        self.labels = torch.as_tensor(labels[:, 0])
        target_shape = (-1, grid_size, grid_size, 28, 28)
        self.images = self.images.reshape(*target_shape)
        self.transform = transform

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, idx: int):
        return self.transform(self.images[idx]), self.labels[idx]


def get_dataloader(grid_size: int, partition: str, batch_size: int):
    download_visudo_dataset(grid_size)
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    train_dataset = SudokuDataset(partition, grid_size, transform=normalize)
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=partition == "train",
    )


class LeNet(nn.Module):
    def __init__(self, nb_classes: int = 10):
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
            nn.Linear(84, nb_classes),
        )
        self.activation = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.encoder(x)
        assert not torch.isnan(x).any()
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        assert not torch.isnan(x).any()
        return self.activation(x)


class VisualSudokuModule(nn.Module):
    def __init__(self, grid_size: int):
        super(VisualSudokuModule, self).__init__()
        self.net = LeNet(grid_size)
        self.circuit = get_circuit(grid_size)
        self.circuit_batched = torch.vmap(self.circuit)
        self.grid_size = grid_size

    def forward(self, images):
        shape = images.shape
        assert not torch.isnan(images).any()
        images = images.reshape(-1, 1, 28, 28)
        image_probs = self.net(images)
        assert not torch.isnan(image_probs).any()
        image_probs = image_probs.reshape(shape[0], -1)
        return self.circuit_batched(image_probs, torch.zeros_like(image_probs))


def get_circuit(grid_size: int):
    circuit = klay.Circuit()
    const_lits = [-x for x in range(1, grid_size**3+1)]
    circuit.add_D4_from_file(f"experiments/visual_sudoku/sudoku_{grid_size}.nnf", true_lits = const_lits)
    print("Nb nodes", circuit.nb_nodes())
    return circuit.to_torch_module()


def nll_loss(preds, targets):
    neg_preds = klay.backends.torch_backend.log1mexp(preds)
    nll = -torch.where(targets, preds, neg_preds)
    return nll.mean()


def main(grid_size: int, batch_size: int, nb_epochs: int, learning_rate: float, device="cuda"):
    train_dataloader = get_dataloader(grid_size, "train", batch_size)
    model = VisualSudokuModule(grid_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.00001)

    for epoch in range(nb_epochs):
        losses = []
        t1 = perf_counter()
        for xs, ys in train_dataloader:
            xs, ys = xs.to(device), ys.to(device)
            preds = model(xs)
            loss = nll_loss(preds[0], ys)
            losses.append(loss.item())
            assert not torch.isnan(loss).any()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch}, Loss {np.mean(losses):.5f} (in {perf_counter() - t1:.2f}s)")

    model = model.eval()
    val_dataloader = get_dataloader(grid_size, "valid", 1)
    accs = []
    for xs, ys in val_dataloader:
        xs, ys = xs.to(device), ys.to(device)
        preds = model(xs).exp()
        acc = (preds[0] > 0.5) == ys
        accs += acc.tolist()
    print(f"Validation Accuracy {np.mean(accs):.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-e', '--nb_epochs', type=int, default=10)
    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0003)
    args = parser.parse_args()

    main(
        grid_size=4,
        batch_size=args.batch_size,
        nb_epochs=args.nb_epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )
