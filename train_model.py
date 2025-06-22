import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Define a simple model
class DigitClassifier(nn.Module):
    def __init__(self):  # ✅ Corrected constructor
        super(DigitClassifier, self).__init__()  # ✅ Corrected super call
        self.layer1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)  # 10 output classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 2. Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Load dataset
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 4. Instantiate model
model = DigitClassifier().to(device)

# 5. Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training loop
for epoch in range(5):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/5], Loss: {total_loss:.4f}")

# 7. Save model
torch.save(model.state_dict(), "trained_model.pth")
print("✅ Model saved as trained_model.pth")
