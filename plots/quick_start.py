import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader

from flashnorm.privacy_engine import FlashNormPrivacyEngine

# Define a simple linear model
class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Create dummy data
X = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=10)

# Define your model, optimizer, and loss function
model = SampleModel()
optimizer = SGD(model.parameters(), lr=0.05)
criterion = nn.MSELoss()

# Initialize PrivacyEngine and make your components private
privacy_engine = FlashNormPrivacyEngine()
model, optimizer, criterion, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    criterion=criterion,
    data_loader=data_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    grad_sample_mode="flash",
)

# Training loop
for X_batch, y_batch in data_loader:
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item()}")