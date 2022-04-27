import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

D_IN = 2
BATCH_SIZE = 32
DATASIZE = 32 * 100
EPOCHS = 100
N_DIGITS = 8


def encode(number):
    def binary_array(digit):
        assert 0 <= digit < 2 ** D_IN
        bin_str = format(digit, "b").zfill(D_IN)
        return np.array(list(bin_str)).astype(float)

    return list(map(binary_array, number))


class MLP(nn.Module):
    def __init__(self, d_in, vocab_size, n_digits):
        super().__init__()
        self.vocab_size = vocab_size
        self.f = nn.Sequential(
            nn.Linear(d_in * n_digits, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, vocab_size * n_digits),
        )

    def forward(self, x):
        x = self.f(x)
        y = torch.softmax(x.view(x.size(0), -1, self.vocab_size), dim=2)
        return y


DIGITS = np.arange(2 ** D_IN)


def random_number(size):
    return np.random.choice(DIGITS, size)


def generate_data(n, size=8):
    return np.array([random_number(size) for _ in range(n)])


def train():
    model = MLP(D_IN, 2 ** D_IN, N_DIGITS)
    data = generate_data(DATASIZE, N_DIGITS)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch}")
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            sorted_batch = np.sort(batch, axis=1)
            source = torch.tensor(list(map(encode, batch))).float()
            source = source.view(BATCH_SIZE, -1)
            preds = model(source)
            y = torch.from_numpy(sorted_batch).flatten()
            preds = preds.view(-1, preds.size(-1))
            acc = torch.sum(preds.argmax(dim=1) == y) / len(y)
            loss = F.cross_entropy(preds, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(f"Loss: {loss}, Acc: {acc.item()}")


train()
