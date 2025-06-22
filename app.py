import streamlit as st
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from model import CVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = CVAE().to(device)
    model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
    model.eval()
    return model

def generate_images(model, digit, n=5):
    digit_tensor = torch.tensor([digit]*n).to(device)
    z = torch.randn(n, model.latent_dim).to(device)
    with torch.no_grad():
        generated = model.decode(z, digit_tensor).cpu()
    return generated

def show_images(images):
    grid = make_grid(images, nrow=len(images), normalize=True, pad_value=1)
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis('off')
    st.pyplot(fig)

st.title("Digit Image Generator")
digit = st.selectbox("Choose a digit", list(range(10)))
if st.button("Generate"):
    model = load_model()
    imgs = generate_images(model, digit)
    show_images(imgs)
# Add this in app.py
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Then in load_model()
def load_model():
    model = DigitClassifier()
    model.load_state_dict(torch.load("trained_model.pth", map_location=device))
    model.eval()
    return model
