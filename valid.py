import argparse
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torchvision import transforms

class SingleNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28), antialias=True),
    transforms.ToTensor(),
])


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = transform(img)
    return tensor.unsqueeze(0)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Путь к изображению цифры")
    args = parser.parse_args()

    model = SingleNeuron().to('cpu')
    state_dict = torch.load("single_neuron_mnist.pth", map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    x = load_image(args.image).to('cpu')

    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1)
        pred = probs.argmax(dim=1).item()

    print(f"Предсказанная цифра: {pred}")
    probs = probs.squeeze().cpu().tolist()
    for digit, prob in enumerate(probs):
        percent = round(prob * 100, 2)
        print(f"Цифра {digit}: {percent:.2f}%")


if __name__ == "__main__":
    main()

