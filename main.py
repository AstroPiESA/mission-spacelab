import math
import os
from time import sleep
import numpy as np

# Import the Camera class from the picamzero (picamera-zero) module
from picamzero import Camera  # type: ignore

# Import OpenCV
import cv2

# Config
image_num = 80
image_interval = 5

iss_height = 408  # km
sensor_width = 6.287  # mm
sensor_height = 4.712  # mm
focal_length = 5  # mm


def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1)
    image_2_cv = cv2.imread(image_2)
    return image_1_cv, image_2_cv


def mask_image(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Filter out white pixels, hsl: 0-179, 0-255, 0-255
    # For hue: 0 is red, 60 is green, 120 is blue
    # We also want to filter out oceans by limiting hue at 90
    lower = np.array([0, 0, 10])
    upper = np.array([90, 255, 185])
    mask = cv2.inRange(hsv_img, lower, upper)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return mask


def apply_filter(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Saturate
    hsv_img[:, :, 1] = cv2.add(hsv_img[:, :, 1], 200)
    saturated_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    # Contrast
    lab_img = cv2.cvtColor(saturated_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(15, 15))
    cl = clahe.apply(l)
    contrast_image = cv2.merge((cl, a, b))
    contrast_image = cv2.cvtColor(contrast_image, cv2.COLOR_LAB2BGR)
    return contrast_image


def calculate_features(image_1_cv, image_2_cv, feature_number, mask_1, mask_2):
    algorithm = cv2.ORB_create(nfeatures=feature_number)
    keypoints_1, descriptors_1 = algorithm.detectAndCompute(image_1_cv, mask_1)
    keypoints_2, descriptors_2 = algorithm.detectAndCompute(image_2_cv, mask_2)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2


def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1, y1) = keypoints_1[image_1_idx].pt
        (x2, y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1, y1))
        coordinates_2.append((x2, y2))
    return coordinates_1, coordinates_2


def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    return all_distances / len(merged_coordinates)


# orbit height is the height of ISS in km
# sensor size is the size of the sensor in mm
# focal length is the focal length of the camera in mm
# img size is the size of the image in pixels
def gsd_calculator(orbit_height, sensor_size, focal_length, img_size):
    return (orbit_height * 1000 * sensor_size) / (focal_length * img_size)


# Create an instance of the Camera class
cam = Camera()

# Capture images at an interval
for i in range(image_num):
    cam.capture_image(f"sequence-{str(i+1).zfill(2)}.jpg")
    sleep(image_interval)

# Average pixel distance change between images
avg_dpx = []
# placeholder for image width and height according to astro pi replay
img_width = 1412
img_height = 1412
for i in range(image_num - 1):
    n = i + 1
    image_1 = f"sequence-{str(n).zfill(2)}.jpg"
    image_2 = f"sequence-{str(n+1).zfill(2)}.jpg"

    # Calculate pixel distance change between two image
    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)
    if i == 0:
        img_width = image_1_cv.shape[1]
        img_height = image_1_cv.shape[0]
    mask_1 = mask_image(image_1_cv)
    mask_2 = mask_image(image_2_cv)
    filtered_image_1 = apply_filter(image_1_cv)
    filtered_image_2 = apply_filter(image_2_cv)
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(
        filtered_image_1, filtered_image_2, 1000, mask_1, mask_2
    )
    matches = calculate_matches(descriptors_1, descriptors_2)  # Match descriptors

    coordinates_1, coordinates_2 = find_matching_coordinates(
        keypoints_1, keypoints_2, matches
    )
    # Average the distance between matching points and add to list of pixel distance changes
    average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    avg_dpx.append(average_feature_distance)

# Calculate ground sampling distance factor
gsd_w = gsd_calculator(iss_height, sensor_width, focal_length, img_width)
gsd_h = gsd_calculator(iss_height, sensor_height, focal_length, img_height)
gsd = max(gsd_w, gsd_h)
avg_dist = gsd * sum(avg_dpx) / len(avg_dpx)
avg_spd = (avg_dist / image_interval) / 1000  # in km/s

# Write average speed to output file
formatted_spd = "{:.4f}".format(avg_spd)
file_path = "result.txt"
with open(file_path, "w") as file:
    file.write(formatted_spd)

# clean up files
for i in range(image_num):
    os.remove(f"sequence-{str(i+1).zfill(2)}.jpg")
