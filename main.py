# Import the Camera class from the picamzero (picamera-zero) module
from picamzero import Camera  # type: ignore
import os

# Make directory to store images
if not os.path.exists("images"):
    os.makedirs("images")

# Create an instance of the Camera class
cam = Camera()

cam.capture_sequence("images/sequence", num_images=3, interval=3)
