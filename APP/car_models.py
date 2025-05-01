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
                 model_path='yolov8n.pt',
                 target_aspect_ratio=1.6):

        self.margin_ratios = margin_ratios or {
            'left': 0.0963302752293578,
            'right': 0.0779816513761468,
            'top': 0.48299319727891155,
            'bottom': 0.4489795918367347
        }
        self.model = YOLO(model_path)
        self.target_aspect_ratio = target_aspect_ratio

    def crop(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, "Error: Could not load image"

        height, width = img.shape[:2]
        results = self.model(img, verbose=False)

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
            return img, "No car detected"

        x1, y1, x2, y2 = largest_bbox
        car_width = x2 - x1
        car_height = y2 - y1

        orig_left_margin = int(self.margin_ratios["left"] * car_width)
        orig_right_margin = int(self.margin_ratios["right"] * car_width)
        top_margin_ratio = max(self.margin_ratios["left"], self.margin_ratios["right"])
        bottom_margin_ratio = top_margin_ratio
        top_margin = int(top_margin_ratio * car_height)
        bottom_margin = int(bottom_margin_ratio * car_height)

        left_margin = orig_left_margin
        right_margin = orig_right_margin

        width_with_margins = car_width + left_margin + right_margin
        height_with_margins = car_height + top_margin + bottom_margin
        initial_ratio = width_with_margins / height_with_margins

        if initial_ratio > self.target_aspect_ratio:
            target_height = int(width_with_margins / self.target_aspect_ratio)
            additional_height = target_height - height_with_margins
            top_margin += additional_height // 2
            bottom_margin += additional_height // 2
            if additional_height % 2 == 1:
                bottom_margin += 1
        else:
            target_width = int(height_with_margins * self.target_aspect_ratio)
            additional_width = target_width - width_with_margins
            left_margin += additional_width // 2
            right_margin += additional_width // 2
            if additional_width % 2 == 1:
                right_margin += 1

        new_x1 = x1 - left_margin
        new_x2 = x2 + right_margin
        new_y1 = y1 - top_margin
        new_y2 = y2 + bottom_margin

        final_width = car_width + left_margin + right_margin
        final_height = car_height + top_margin + bottom_margin
        final_ratio = final_width / final_height
        print(f"Car dimensions: {car_width}x{car_height}")
        print(f"Margins - Left: {left_margin}, Right: {right_margin}, Top: {top_margin}, Bottom: {bottom_margin}")
        print(
            f"Final dimensions: {final_width}x{final_height}, Ratio: {final_ratio:.3f} (target: {self.target_aspect_ratio})")

        exceeds_left = new_x1 < 0
        exceeds_right = new_x2 > width
        exceeds_top = new_y1 < 0
        exceeds_bottom = new_y2 > height
        exceeds_boundary = exceeds_left or exceeds_right or exceeds_top or exceeds_bottom

        if not exceeds_boundary:
            cropped_img = img[new_y1:new_y2, new_x1:new_x2]
            return cropped_img, "Success"
        else:
            expanded_width = width + max(0, -new_x1) + max(0, new_x2 - width)
            expanded_height = height + max(0, -new_y1) + max(0, new_y2 - height)
            expanded_img = np.ones((expanded_height, expanded_width, 3), dtype=np.uint8)
            expanded_img[:, :] = (128, 128, 128)

            left_offset = max(0, -new_x1)
            top_offset = max(0, -new_y1)
            expanded_img[top_offset:top_offset + height, left_offset:left_offset + width] = img

            adjusted_x1 = int(x1 + left_offset)
            adjusted_y1 = int(y1 + top_offset)
            adjusted_x2 = int(x2 + left_offset)
            adjusted_y2 = int(y2 + top_offset)
            cv2.rectangle(expanded_img, (adjusted_x1, adjusted_y1), (adjusted_x2, adjusted_y2), (255, 0, 0), 2)

            desired_x1 = int(max(0, new_x1) + left_offset)
            desired_y1 = int(max(0, new_y1) + top_offset)
            desired_x2 = int(min(width, new_x2) + left_offset)
            desired_y2 = int(min(height, new_y2) + top_offset)
            cv2.rectangle(expanded_img, (desired_x1, desired_y1), (desired_x2, desired_y2), (0, 0, 255), 2)

            final_crop_x1 = int(left_offset + x1 - left_margin)
            final_crop_y1 = int(top_offset + y1 - top_margin)
            final_crop_x2 = int(final_crop_x1 + final_width)
            final_crop_y2 = int(final_crop_y1 + final_height)

            result_img = expanded_img[final_crop_y1:final_crop_y2, final_crop_x1:final_crop_x2]

            result_ratio = result_img.shape[1] / result_img.shape[0]
            print(f"Final image aspect ratio: {result_ratio:.3f} (target: {self.target_aspect_ratio})")

            expanded_sides = []
            if exceeds_left:
                expanded_sides.append("the left")
            if exceeds_right:
                expanded_sides.append("the right")
            if exceeds_top:
                expanded_sides.append("the top")
            if exceeds_bottom:
                expanded_sides.append("the bottom")

            status = "Make more space to " + ", to ".join(expanded_sides)
            return result_img, status
