import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from numpy import argmax

torch.classes.__path__ = []


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        IMAGE_SIZE = 128
        super(SimpleCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class CarSide:
    classes = [
        "angle-back-on-left",
        "angle-back-on-right",
        "angle-front-on-left",
        "angle-front-on-right",
        "back",
        "front",
        "profile-on-left",
        "profile-on-right",
    ]

    def __init__(self, path):
        self.IMAGE_SIZE = 128
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )
        self.model = SimpleCNN(num_classes=8)
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()

    def predict(self, pil_image):
        x = self.transform(pil_image)
        x.unsqueeze_(0)
        prediction = self.model(x)
        return CarSide.classes[argmax(prediction.detach().numpy()[0])]


class CarOrNot:
    classes = ["car", "not_car"]

    def __init__(self, path):
        self.IMAGE_SIZE = 128
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )
        self.model = SimpleCNN(num_classes=2)
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()

    def predict(self, pil_image):
        x = self.transform(pil_image)
        x.unsqueeze_(0)
        prediction = self.model(x)
        return CarOrNot.classes[argmax(prediction.detach().numpy()[0])]
