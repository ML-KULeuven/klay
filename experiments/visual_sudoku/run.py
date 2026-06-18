import argparse
from pathlib import Path
from time import perf_counter

import klay
import kompyle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class SudokuDataset(Dataset):
    def __init__(self, partition: str, grid_size: int = 4, transform=None):
        super().__init__()
        data_path = Path(__file__).parent / f"visudo{grid_size}"
        features_file = data_path / f'{partition}_puzzle_pixels.txt'
        labels_file = data_path / f'{partition}_puzzle_labels.txt'
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
        super().__init__()
        self.net = LeNet(grid_size)
        self.circuit = get_circuit(grid_size)
        self.grid_size = grid_size

    def forward(self, images):
        shape = images.shape
        images = images.reshape(-1, 1, 28, 28)
        image_probs = self.net(images)
        assert not torch.isnan(image_probs).any()
        image_probs = image_probs.reshape(shape[0], -1)
        return self.circuit(image_probs, torch.zeros_like(image_probs))


def get_circuit(grid_size: int):
    circuit = kompyle.Circuit()
    cnf_file = f"experiments/visual_sudoku/sudoku_{grid_size}.cnf"
    root = kompyle.compile_from_cnf_using_d4v2(circuit, cnf_file)
    circuit.set_root(root)
    print("Nb nodes", circuit.nb_nodes())
    return circuit.to_torch_module()


def nll_loss(preds, targets):
    neg_preds = klay.torch.utils.log1mexp(preds)
    nll = -torch.where(targets, preds, neg_preds)
    return nll.mean()


def train(model, optimizer, dataloader, device="cuda"):
    losses = []
    for xs, ys in dataloader:
        xs, ys = xs.to(device), ys.to(device)
        preds = model(xs)
        loss = nll_loss(preds[0], ys)
        losses.append(loss.item())
        assert not torch.isnan(loss).any()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        optimizer.zero_grad()
    return losses


def evaluate(model, dataloader, device="cuda"):
    model = model.eval()
    accs = []
    for xs, ys in dataloader:
        xs, ys = xs.to(device), ys.to(device)
        preds = model(xs).exp()
        acc = (preds[0] > 0.5) == ys
        accs += acc.tolist()
    return accs


def main(grid_size: int, batch_size: int, nb_epochs: int, learning_rate: float, device="cuda"):
    train_dataloader = get_dataloader(grid_size, "train", batch_size)
    model = VisualSudokuModule(grid_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    timings = []

    for epoch in range(nb_epochs):
        t1 = perf_counter()
        losses = train(model, optimizer, train_dataloader, device)
        timings.append(perf_counter() - t1)
        print(f"Epoch {epoch}, Loss {np.mean(losses):.6f}")
    print(f"Mean Epoch Time (s) {np.mean(timings):.3f} ± {np.std(timings):.3f}")

    val_dataloader = get_dataloader(grid_size, "valid", 1)
    accs = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy {100*np.mean(accs):.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-e', '--nb_epochs', type=int, default=20)
    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    main(
        grid_size=4,
        batch_size=args.batch_size,
        nb_epochs=args.nb_epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )
