import os
import sys
sys.path.append("..")
from environment import CarlaEnvironment  # noqa

IMAGE_HEIGHT = 300
IMAGE_WIDTH = 533


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
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])

    def canny(self, image):
        print(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny1 = cv2.Canny(blur, 50, 150)  # low and high threshold
        return canny1

    def display_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image

    def region_of_interest(self, image):
        polygons = np.array([[(0, 300), (533, 300), (100, 300)]])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_or(image, mask)
        return masked_image


def main():
    env = CarlaEnvironment(IMAGE_HEIGHT, IMAGE_WIDTH)
    detector = LaneDetector(IMAGE_HEIGHT, IMAGE_WIDTH)
    env.lane_detection_vehicle()
    env.add_rgb_camera()
    frame = env.rgb_camera_data

    while True:
        frame = env.rgb_camera_data
        count += 1
        canny_image = detector.canny(frame)
        cropped_image = detector.region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)  # noqa
        averaged_lines = detector.averaged_slope_intercept(frame, lines)
        line_image = detector.display_lines(frame, averaged_lines)
        merged_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("result", merged_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    env.cleanup()


if __name__ == "__main__":
    main()
