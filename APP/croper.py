import cv2
from ultralytics import YOLO
import os

# model = YOLO("../croper/yolov8n.pt")


class Croper:
    def __init__(self):
        self.model = YOLO("./yolov8n.pt")

    def get_croped(self, img):
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

        if largest_bbox:
            x1, y1, x2, y2 = largest_bbox
            cropped_img = img[y1:y2, x1:x2]
            # cv2.imwrite(output_path, cropped_img)
            return cropped_img
        else:
            # cv2.imwrite(output_path, img)
            return False


# def crop_cars(image_path, output_path):

#     img = cv2.imread(image_path)
#     results = model(img, verbose=False)

#     largest_area = 0
#     largest_bbox = None

#     for bbox, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
#         if int(cls) == 2:
#             x1, y1, x2, y2 = map(int, bbox)
#             area = (x2 - x1) * (y2 - y1)
#             if area > largest_area:
#                 largest_area = area
#                 largest_bbox = (x1, y1, x2, y2)

#     if largest_bbox:
#         x1, y1, x2, y2 = largest_bbox
#         cropped_img = img[y1:y2, x1:x2]
#         # cv2.imwrite(output_path, cropped_img)
#         return True
#     else:
#         # cv2.imwrite(output_path, img)
#         return False
