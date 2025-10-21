import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ==========================
# 1. Load and preprocess data
# ==========================
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create character mappings
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)
print(f'Total characters: {len(text)}, Unique characters: {vocab_size}')

# Prepare input and target sequences
seq_len = 100  # Sequence length
step = 1
X = []
y = []

for i in range(0, len(text) - seq_len, step):
    X.append([char_to_idx[ch] for ch in text[i:i+seq_len]])
    y.append([char_to_idx[ch] for ch in text[i+1:i+seq_len+1]])

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)
print(f'Total sequences: {X.shape[0]}')

# ==========================
# 2. Define LSTM model
# ==========================
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=2):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden states with zeros
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                torch.zeros(self.n_layers, batch_size, self.hidden_size))

# ==========================
# 3. Training
# ==========================
hidden_size = 256
n_layers = 2
batch_size = 1  # For simplicity

model = CharLSTM(vocab_size, hidden_size, vocab_size, n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

n_epochs = 10  # You can increase to 50+ for better text generation

for epoch in range(n_epochs):
    hidden = model.init_hidden(batch_size)
    total_loss = 0
    for i in range(X.shape[0]):
        optimizer.zero_grad()
        input_seq = X[i].unsqueeze(0)   # Add batch dimension
        target_seq = y[i].unsqueeze(0)
        output, hidden = model(input_seq, hidden)
        hidden = (hidden[0].detach(), hidden[1].detach())  # Detach hidden state
        loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 5000 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} Average Loss: {total_loss / X.shape[0]:.4f}")

# ==========================
# 4. Text generation
# ==========================
def generate_text(model, start_text, length=500):
    model.eval()
    hidden = model.init_hidden(1)
    input_seq = torch.tensor([char_to_idx[ch] for ch in start_text], dtype=torch.long).unsqueeze(0)
    generated = list(start_text)

    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        probs = torch.softmax(output[0, -1], dim=0).detach().numpy()
        char_idx = np.random.choice(range(vocab_size), p=probs)
        generated.append(idx_to_char[char_idx])
        input_seq = torch.tensor([[char_idx]], dtype=torch.long)

    return ''.join(generated)

# Example usage
start_prompt = "To be or not to be, "
generated_text = generate_text(model, start_prompt, length=500)
print("Generated Text:\n")
print(generated_text)
