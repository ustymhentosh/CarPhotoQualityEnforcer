import torch
from PIL import Image
import cv2
from torchvision import transforms
import torch.nn as nn
from numpy import argmax
from ultralytics import YOLO
import numpy as np

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

class OverexposedOrNot:
    classes = ["not_overexposed", "overexposed"]

    def __init__(self, path):
        self.IMAGE_SIZE = 128
        self.transform = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor()
        ])
        self.model = SimpleCNN(num_classes=2)
        self.model.load_state_dict(torch.load('overexposed_model.pth', map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, pil_image):
        x = self.transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(x)
            pred = (output > 0.5).item()
        return OverexposedOrNot.classes[int(pred)]

class CarCropper:
    def __init__(self,
                 margin_ratios=None,
                 model=None,
                 target_aspect_ratio=1.6,
                 acceptance_limit=0.05):
        if margin_ratios is None:
            margin_ratios = {'left': 0.0963302752293578,
                             'right': 0.0779816513761468,
                             'top': 0.48299319727891155,
                             'bottom': 0.4489795918367347}
        self.acceptance_limit = acceptance_limit
        self.margin_ratios = margin_ratios
        self.target_aspect_ratio = target_aspect_ratio
        self.model = model if model is not None else YOLO('yolov8n.pt')

    def crop(self, pil_image):
        # Convert PIL image to OpenCV format (numpy array, BGR)
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]

        results = self.model(img, verbose=False)

        # Find largest car (YOLO class 2)
        largest_area = 0
        largest_bbox = None

        for bbox, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            if int(cls) == 2:
                x1, y1, x2, y2 = map(int, bbox)
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_bbox = (x1, y1, x2, y2)

        if not largest_bbox:
            return pil_image, "No car detected"

        x1, y1, x2, y2 = largest_bbox
        car_width = x2 - x1
        car_height = y2 - y1

        # Calculate margins
        left_margin = int(self.margin_ratios["left"] * car_width)
        right_margin = int(self.margin_ratios["right"] * car_width)
        vert_margin_ratio = max(self.margin_ratios["left"], self.margin_ratios["right"])
        top_margin = int(vert_margin_ratio * car_height)
        bottom_margin = int(vert_margin_ratio * car_height)

        width_with_margins = car_width + left_margin + right_margin
        height_with_margins = car_height + top_margin + bottom_margin
        initial_ratio = width_with_margins / height_with_margins

        # Adjust margins for aspect ratio
        if initial_ratio > self.target_aspect_ratio:
            target_height = int(width_with_margins / self.target_aspect_ratio)
            extra = target_height - height_with_margins
            top_margin += extra // 2
            bottom_margin += extra // 2 + (extra % 2)
        else:
            target_width = int(height_with_margins * self.target_aspect_ratio)
            extra = target_width - width_with_margins
            left_margin += extra // 2
            right_margin += extra // 2 + (extra % 2)

        new_x1 = x1 - left_margin
        new_x2 = x2 + right_margin
        new_y1 = y1 - top_margin
        new_y2 = y2 + bottom_margin

        # Check boundary exceedance
        pad_threshold_x = int(self.acceptance_limit * width)
        pad_threshold_y = int(self.acceptance_limit * height)

        exceeds_left = -new_x1 > pad_threshold_x
        exceeds_right = (new_x2 - width) > pad_threshold_x
        exceeds_top = -new_y1 > pad_threshold_y
        exceeds_bottom = (new_y2 - height) > pad_threshold_y

        exceeds_critically = exceeds_left or exceeds_right or exceeds_top or exceeds_bottom

        # Clamp coordinates to image size
        crop_x1 = max(0, new_x1)
        crop_y1 = max(0, new_y1)
        crop_x2 = min(width, new_x2)
        crop_y2 = min(height, new_y2)

        cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

        # Convert back to PIL for consistency
        pil_cropped = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

        if exceeds_critically:
            status = "Make more space to " + ", to ".join([
                side for side, flag in zip(
                    ["the left", "the right", "the top", "the bottom"],
                    [exceeds_left, exceeds_right, exceeds_top, exceeds_bottom]
                ) if flag
            ])
        else:
            status = "Success"

        return pil_cropped, status

