import math
import numpy as np

# Import the Camera class from the picamzero (picamera-zero) module
from picamzero import Camera  # type: ignore

# Import OpenCV
import cv2


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
    upper = np.array([90, 255, 165])
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


def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(
        image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None
    )
    resize = cv2.resize(match_img, (1200, 600), interpolation=cv2.INTER_AREA)
    cv2.imshow("matches", resize)
    cv2.waitKey(0)
    cv2.destroyWindow("matches")


# Create an instance of the Camera class
cam = Camera()

cam.capture_sequence("sequence", num_images=3, interval=3)

image_1 = "sequence-1.jpg"
image_2 = "sequence-2.jpg"

image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)
mask_1 = mask_image(image_1_cv)
mask_2 = mask_image(image_2_cv)
filtered_image_1 = apply_filter(image_1_cv)
filtered_image_2 = apply_filter(image_2_cv)
keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(
    filtered_image_1, filtered_image_2, 1000, mask_1, mask_2
)  # Get keypoints and descriptors
matches = calculate_matches(descriptors_1, descriptors_2)  # Match descriptors
display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches)
