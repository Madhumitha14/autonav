import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from environment import CarlaEnvironment  # noqa

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280


class LaneDetector:
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width

    def make_coordinates(self, image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def averaged_slope_intercept(self, image, lines):
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = self.make_coordinates(image, left_fit_average)
        right_line = self.make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])

    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny1 = cv2.Canny(blur, 50, 150)  # low and high threshold
        return canny1

    def display_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 0), 10)
        return line_image

    def region_of_interest(self, image):
        height = image.shape[0]
        polygons = np.array([[(100, height), (1000, height), (700, 320)]])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image


def main(town):
    env = CarlaEnvironment(IMAGE_HEIGHT, IMAGE_WIDTH, town)
    detector = LaneDetector(IMAGE_HEIGHT, IMAGE_WIDTH)
    env.lane_detection_vehicle()
    env.add_rgb_camera(1, 0, 2)
    frame = env.rgb_camera_data
    time.sleep(5)
    start_time = time.time()

    while time.time() - start_time < 600:
        frame = env.rgb_camera_data
        try:
            canny_image = detector.canny(frame)
            cv2.imwrite(os.path.join("images", "canny", f"{int(time.time())}.jpeg"), canny_image)
            cv2.waitKey(1)
            cropped_image = detector.region_of_interest(canny_image)
            cv2.imwrite(os.path.join("images", "cropped", f"{int(time.time())}.jpeg"), cropped_image)
            cv2.waitKey(1)
            lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)  # noqa
            averaged_lines = detector.averaged_slope_intercept(frame, lines)
            line_image = detector.display_lines(frame, averaged_lines)
            cv2.imwrite(os.path.join("images", "lines", f"{int(time.time())}.jpeg"), line_image)
            cv2.waitKey(1)
            merged_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        except:
            merged_image = frame

        cv2.imwrite(os.path.join("images", "result", f"{int(time.time())}.jpeg"), merged_image)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    env.cleanup()



if __name__ == "__main__":
    main("Town03")
