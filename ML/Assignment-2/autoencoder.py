import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load built-in MNIST dataset
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor()  # Converts images to [0,1] tensors
])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split into train/test
train_size = int(0.8 * len(mnist_dataset))
test_size = len(mnist_dataset) - train_size
train_ds, test_ds = random_split(mnist_dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# -----------------------------
# 2. Define Autoencoder
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(28*28, latent_dim)
        self.decoder = nn.Linear(latent_dim, 28*28)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28*28)
        latent = self.sigmoid(self.encoder(x))
        recon = self.sigmoid(self.decoder(latent))
        return recon

# -----------------------------
# 3. Train Autoencoder
# -----------------------------
def train_autoencoder(latent_dim, epochs=10, lr=0.001):
    model = Autoencoder(latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for data, _ in train_loader:
            img = data
            optimizer.zero_grad()
            recon = model(img)
            loss = criterion(recon, img.view(-1, 28*28))
            loss.backward()
            optimizer.step()
    
    # Evaluate reconstruction MSE on test set
    model.eval()
    mse = 0
    with torch.no_grad():
        for data, _ in test_loader:
            img = data
            recon = model(img)
            mse += criterion(recon, img.view(-1, 28*28)).item() * len(img)
    mse /= len(test_loader.dataset)
    return mse, model

# -----------------------------
# 4. Run for different latent sizes
# -----------------------------
latent_sizes = [2, 10, 50]
mses = {}
models = {}

for size in latent_sizes:
    mse, model = train_autoencoder(size, epochs=10, lr=0.001)
    mses[size] = mse
    models[size] = model
    print(f"Latent Size {size} -> Test MSE: {mse:.6f}")

# -----------------------------
# 5. Visualize reconstructions
# -----------------------------
def show_reconstruction(model, num_images=5):
    model.eval()
    data_iter = iter(test_loader)
    imgs, _ = next(data_iter)
    with torch.no_grad():
        recon = model(imgs)
    imgs = imgs[:num_images]
    recon = recon.view(-1, 1, 28, 28)[:num_images]

    fig, axes = plt.subplots(2, num_images, figsize=(num_images*2,4))
    for i in range(num_images):
        axes[0,i].imshow(imgs[i].squeeze(), cmap='gray')
        axes[0,i].axis('off')
        axes[1,i].imshow(recon[i].squeeze(), cmap='gray')
        axes[1,i].axis('off')
    axes[0,0].set_title("Original")
    axes[1,0].set_title("Reconstructed")
    plt.show()

# Show reconstruction for each latent size
for size in latent_sizes:
    print(f"Reconstruction with Latent Size = {size}")
    show_reconstruction(models[size])
