# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
from PIL import Image
from torchvision import transforms
from typing import Any, Callable, Type


# %%
# Set default tensor to gpu
def set_default_tensor_type():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    return device


# %%
def train(
    net: Type[nn.Module],
    img: torch.tensor,
    loss_fn: Callable[[Any, Any], torch.float],
    num_epochs: int,
    lr: float = 0.001,
    use_adam: bool = False,
    mask=None,
) -> tuple[Type[nn.Module], list[float]]:
    f = net()
    opt = Adam(f.parameters(), lr=lr) if use_adam else SGD(f.parameters(), lr=lr)
    losses = []
    coords = torch.stack(
        [
            torch.tensor([i / img.shape[0], j / img.shape[1]], dtype=torch.float32)
            for i, j in np.ndindex(img.shape[:2])
        ]
    )  # R^{x * y, 2}

    target = img.reshape(-1, 3)  # R^{x * y, 3}

    # remove coordinates of the masked pixels if mask is provided
    if mask is not None:
        mask_flat = mask.flatten()
        mask_flat.shape
        coords = coords[~mask_flat]  # R^{(x * y - m), 2}
        target = target[~mask_flat]

    for ep in range(num_epochs):
        opt.zero_grad()
        y_hat = f(
            coords
        )  # Forward pass through the network for all coordinates at once
        loss = loss_fn(y_hat, target)
        loss.backward()
        opt.step()
        if ep % 100 == 0:
            print(f"Epoch {ep+1}/{num_epochs}, Loss: {loss.item()}")
        losses.append(loss.item())

    return f, losses


# %%
def positional_encoding(xys: torch.Tensor, l_max=10) -> torch.Tensor:
    xys = xys.float()  # Ensure the inputs are floats for trigonometric operations
    pi = torch.pi
    frequencies = 2 ** torch.arange(l_max) * pi  # Calculate frequency levels

    # Preallocate the result tensor
    results = torch.zeros(xys.size(0), 4 * l_max)

    # Calculate sine and cosine for each frequency and stack them
    for i, freq in enumerate(frequencies):
        results[:, 4 * i + 0] = torch.sin(xys[:, 0] * freq)
        results[:, 4 * i + 1] = torch.cos(xys[:, 0] * freq)
        results[:, 4 * i + 2] = torch.sin(xys[:, 1] * freq)
        results[:, 4 * i + 3] = torch.cos(xys[:, 1] * freq)

    return results


# %%
class NerfNetworkEncoded(nn.Module):
    def __init__(self, num_frequencies: int = 10):
        super(NerfNetworkEncoded, self).__init__()
        self.num_frequencies = num_frequencies
        self.fc1 = nn.Linear(4 * self.num_frequencies, 256, dtype=torch.float32)
        self.fc2 = nn.Linear(256, 256, dtype=torch.float32)
        self.fc3 = nn.Linear(256, 256, dtype=torch.float32)
        self.fc4 = nn.Linear(256, 256, dtype=torch.float32)
        self.fc5 = nn.Linear(256, 3, dtype=torch.float32)
        self.R = nn.ReLU()

    def forward(self, x):
        x = positional_encoding(x, self.num_frequencies)
        x = self.R(self.fc1(x))
        x = self.R(self.fc2(x))
        x = self.R(self.fc3(x))
        x = self.R(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x


# %%
def inpaint_nerf(
    image: np.array,
    mask: np.array,
    device: torch.device,
    epochs: int = 500,
    l=10,
) -> np.array:
    image_th = torch.tensor(image, dtype=torch.float32).to(device)
    mask_th = torch.tensor(mask, dtype=torch.bool).to(device)

    loss_fn = nn.MSELoss()
    f, losses = train(
        lambda: NerfNetworkEncoded(l),
        image_th,
        loss_fn,
        epochs,
        lr=0.001,
        use_adam=True,
        mask=mask_th,
    )
    coords = torch.stack(
        [
            torch.tensor([i / image.shape[0], j / image.shape[1]], dtype=torch.float32)
            for i, j in np.ndindex(mask.shape[:2])
        ]
    )
    y_hat = f(coords).cpu().detach().numpy().reshape(mask.shape + (3,))
    inpainted_image = np.zeros(image.shape)
    inpainted_image[mask == False] = image[mask == False]
    inpainted_image[mask == True] = y_hat[mask == True]

    return inpainted_image, losses
