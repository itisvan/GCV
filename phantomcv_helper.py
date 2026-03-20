import cv2
import gtuner
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import random
import settings
from ultralytics import YOLO

class Phantom:
    def __init__(self):
        # Initialize the Phantom object with settings from the settings module.
        self.confidence = settings.confidence
        self.IoUThreshold = settings.IoUThreshold
        self.speedX = settings.speedX
        self.speedY = settings.speedY
        self.aimSmoothing = settings.aimSmoothing
        self.manualOveride = settings.manualOveride
        self.dataCollectionMode = settings.dataCollectionMode
        self.boundingBoxX1 = settings.boundingBoxX1
        self.boundingBoxX2 = settings.boundingBoxX2
        self.boundingBoxY1 = settings.boundingBoxY1
        self.boundingBoxY2 = settings.boundingBoxY2
        self.showBoundingBox = settings.showBoundingBox
        self.color = [settings.boundingBoxColor[0], settings.boundingBoxColor[1], settings.boundingBoxColor[2]]
        self.adaptiveBoundingBox = settings.adaptiveBoundingBox
        self.currentFrame = 0
        self.model = None

    @staticmethod
    def seed_everything(seed):
        # Set the seeds for all relevant libraries for reproducible results.
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def load_model(self, modelName):
        # Load a YOLOv12 model using the Ultralytics unified API.
        SEED = 42
        Phantom.seed_everything(SEED)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), modelName + '.pt')
        self.model = YOLO(model_path)
        return self.model

    @staticmethod
    def rectangleScaling(x1, y1, x2, y2):
        # Calculate the headshot aim point (upper third of bounding box center).
        distanceX = abs((x1 - x2)) / 2
        distanceY = abs((y1 - y2)) / 3
        centerX = int(distanceX) + x1
        headshotY = int(distanceY) + y1
        return (centerX, headshotY)

    def draw_border(self, img, pt1, pt2, thickness, r, d):
        # Draw a rounded-corner border around a rectangular region in an image.
        x1, y1 = pt1
        x2, y2 = pt2

        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), self.color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), self.color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, self.color, thickness)

        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), self.color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), self.color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, self.color, thickness)

        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), self.color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), self.color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, self.color, thickness)

        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), self.color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), self.color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, self.color, thickness)
        return img

    def predict(self, frame, button_5, button_8, button_9):
        # Process an input frame with YOLOv12, run object detection, and return
        # the processed frame and right stick coordinates for aiming.
        rx, ry = 0, 0
        final = frame
        x1, y1, x2, y2 = 960, 540, 960, 540
        X1, Y1, X2, Y2 = self.boundingBoxX1, self.boundingBoxY1, self.boundingBoxX2, self.boundingBoxY2

        if self.adaptiveBoundingBox and button_8 > 0:
            X1, Y1, X2, Y2 = 400, 180, 1500, 900

        img0 = frame[Y1:Y2, X1:X2, :]

        if self.showBoundingBox:
            img_bounding = self.draw_border(frame, (x1, y1), (x2, y2), 8, 15, 30)
            img_bounding = self.draw_border(img_bounding, (0, 0), (1920, 1080), 8, 15, 30)
        else:
            img_bounding = frame

        # Run YOLOv12 inference — imgsz=416 kept for consistency with original model training
        results = self.model(img0, imgsz=416, verbose=False,
                             conf=self.confidence, iou=self.IoUThreshold,
                             agnostic_nms=True, max_det=10)

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0]]
                label = f"Player {conf:.2f}"

                if cls == 0:  # Player detected
                    bx1, by1, bx2, by2 = (bx1 + X1), (by1 + Y1), (bx2 + X1), (by2 + Y1)
                    rx, ry = self.trajectory(bx1, by1, bx2, by2)

                    if self.manualOveride and button_9 > 0:
                        rx, ry = 0, 0
                    print(str(rx) + ' ' + str(ry))

                    if button_5 > 0:
                        if self.dataCollectionMode:
                            print('Player Detected on Screen! Skipping Frame')

                    if self.showBoundingBox:
                        draw1 = self.draw_border(img_bounding, (bx1, by1), (bx2, by2), 4, 15, 30)
                        draw2 = cv2.putText(draw1, label, (bx1, by1 - 20), cv2.FONT_HERSHEY_PLAIN, 2, self.color, 2)
                        draw3 = cv2.putText(draw2, str(Phantom.rectangleScaling(bx1, by1, bx2, by2)), (bx1, by1 - 50), cv2.FONT_HERSHEY_PLAIN, 2, self.color, 2)
                        draw4 = cv2.circle(draw3, Phantom.rectangleScaling(bx1, by1, bx2, by2), 15, (0, 0, 255), -1)
                        final = cv2.line(draw4, (960, 540), Phantom.rectangleScaling(bx1, by1, bx2, by2), (0, 0, 255), 3)

                elif self.dataCollectionMode:
                    if button_5 > 0:
                        self.save_data(img0)

        return (final, rx, ry)

    def save_data(self, img_path):
        # Save the current frame as an image file for data collection.
        print('Player Not Detected!')
        name = './data/frame' + str(self.currentFrame) + '.jpg'
        print('Creating...' + name)
        cv2.imwrite(name, img_path)
        self.currentFrame += 1

    def trajectory(self, x1, y1, x2, y2):
        # Calculate the trajectory required to move the aim from the current position to the target position.
        rx, ry = 0, 0
        crosshairPositionX, crosshairPositionY = 960, 540
        playerCoordX, playerCoordY = Phantom.rectangleScaling(x1, y1, x2, y2)
        distance = (((playerCoordX - crosshairPositionX) ** 2) + ((playerCoordY - crosshairPositionY) ** 2)) ** 0.5

        deltaX = playerCoordX - crosshairPositionX
        deltaY = playerCoordY - crosshairPositionY

        pForceX = deltaX / 1920
        pForceY = deltaY / 1080

        dX = (pForceX * self.speedX) * 100
        dY = (pForceY * self.speedY) * 100

        if self.aimSmoothing:
            dX, dY = self.apply_aim_smoothing(distance, dX, dY)

        rx, ry = dX, dY
        return (rx, ry)

    @staticmethod
    def apply_aim_smoothing(distance, rx, ry):
        # Apply smoothing to the calculated trajectory based on distance to target.
        speed = 1.171429 + (5.851389e-20 * distance) + (0.000002380952 * distance ** 2)
        return (rx * speed), (ry * speed)

    def process(self, frame, gcvdata):
        ### Main processing function: get controller button states, process the frame
        ### using the predict function, and update gcvdata with right stick coordinates.
        button_5 = gtuner.get_actual(gtuner.BUTTON_5)  # RT
        button_8 = gtuner.get_actual(gtuner.BUTTON_8)  # LT
        button_9 = gtuner.get_actual(gtuner.BUTTON_9)  # LS

        ### PREDICT AND MODIFY FRAMES ###
        frame = cv2.putText(frame, "Phantom CV v0.8 (YOLOv12)", (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, (102, 51, 153), 2)
        frame_process, rightStickX, rightStickY = self.predict(frame, button_5, button_8, button_9)
        frame_process = cv2.circle(frame_process, (960, 540), 70, self.color, 2)

        ### SEND DATA TO GPC SCRIPT FOR INTERPRETATION ###
        gcvdata.extend(int(float(rightStickX) * 0x10000).to_bytes(4, byteorder="big", signed=True))
        gcvdata.extend(int(float(rightStickY) * 0x10000).to_bytes(4, byteorder="big", signed=True))
        return frame_process, gcvdata


def PyInit_phantomcv_helper():
    return None
