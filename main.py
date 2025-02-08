import math

# Import the Camera class from the picamzero (picamera-zero) module
from picamzero import Camera  # type: ignore

# Import OpenCV
import cv2


def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv


def calculate_features(image_1_cv, image_2_cv, feature_number):
    algorithm = cv2.ORB_create(nfeatures=feature_number)
    keypoints_1, descriptors_1 = algorithm.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = algorithm.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2


def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(
    image_1_cv, image_2_cv, 1000
)  # Get keypoints and descriptors
matches = calculate_matches(descriptors_1, descriptors_2)  # Match descriptors
display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches)
