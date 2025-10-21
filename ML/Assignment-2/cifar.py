import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load CIFAR-10 dataset
# -----------------------------
(X_train, _), (X_test, _) = cifar10.load_data()  # Labels not needed for autoencoder

# Normalize to [0,1] and reshape to (num_samples, 32*32*3)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1, 32*32*3)
X_test = X_test.reshape(-1, 32*32*3)

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# 2. Define Autoencoder
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(3072, latent_dim)
        self.decoder = nn.Linear(latent_dim, 3072)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 3072)
        latent = self.relu(self.encoder(x))
        recon = self.sigmoid(self.decoder(latent))
        return recon

# -----------------------------
# 3. Train Autoencoder
# -----------------------------
model = Autoencoder(latent_dim=128)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for data in train_loader:
        img = data[0]
        optimizer.zero_grad()
        recon = model(img)
        loss = criterion(recon, img.view(-1, 3072))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.6f}")

# -----------------------------
# 4. Evaluate reconstruction MSE
# -----------------------------
model.eval()
mse = 0
with torch.no_grad():
    for data in test_loader:
        img = data[0]
        recon = model(img)
        mse += criterion(recon, img.view(-1, 3072)).item() * len(img)
mse /= len(test_loader.dataset)
print(f"Test MSE: {mse:.6f}")

# -----------------------------
# 5. Visualize original vs reconstructed images
# -----------------------------
def show_reconstruction(model, num_images=5):
    model.eval()
    data_iter = iter(test_loader)
    imgs = next(data_iter)[0][:num_images]
    
    with torch.no_grad():
        recon = model(imgs).view(-1, 3, 32, 32).permute(0,2,3,1)  # Shape: (N,32,32,3)
        imgs = imgs.view(-1, 3, 32, 32).permute(0,2,3,1)
    
    fig, axes = plt.subplots(2, num_images, figsize=(num_images*2,4))
    for i in range(num_images):
        axes[0,i].imshow(imgs[i].numpy())
        axes[0,i].axis('off')
        axes[1,i].imshow(recon[i].numpy())
        axes[1,i].axis('off')
    axes[0,0].set_title("Original")
    axes[1,0].set_title("Reconstructed")
    plt.show()

show_reconstruction(model)
