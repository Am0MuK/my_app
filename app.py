# Import the libraries
import cv2 # For image processing
import pytesseract # For OCR
import numpy as np # For array manipulation
import pandas as pd # For data frame manipulation
from flask import Flask, render_template, request, redirect, url_for # For web app

app = Flask(__name__) # Create a flask app instance

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'} # Define the allowed image file extensions

def detect_text_region(img):
    """Detect the region of the image containing text"""
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(max_contour)

    # Expand the bounding box slightly to include any text that may be close to the edges
    roi_x = max(0, x - 10)
    roi_y = max(0, y - 10)
    roi_w = min(img.shape[1] - roi_x, w + 20)
    roi_h = min(img.shape[0] - roi_y, h + 20)

    return roi_x, roi_y, roi_w, roi_h

OCR_REGION = detect_text_region(img)


lang = "ron" # Define the language for OCR as romanian
config = "--psm 6" # Define the page segmentation mode as 6 (assume a single uniform block of text)
config += " --oem 1" # Define the OCR engine mode as 1 (use LSTM only)
config += " -c tessedit_char_whitelist=AĂăÂâBbCcDdEeFfGgHhIiÎîJjKkLlMmNnOoPpQqRrSsȘșTtȚțUuVvWwXxYyZz1234567890.-,"
config += " -c tessedit_char_blacklist=абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"


def allowed_file(filename):
    """Check if a file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    """Preprocess an image for OCR"""
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale

    blur = cv2.GaussianBlur(gray, (5,5), 0) # Apply a Gaussian blur to reduce noise

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) # Apply an adaptive threshold to binarize the image

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # Create a rectangular kernel for morphological operations
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # Apply a closing operation to fill small gaps and holes

    return morph # Return the preprocessed image

def extract_text(img):
    """Extract text from an image"""
   
    img = preprocess_image(img) # Preprocess the image

    x, y, w, h = OCR_REGION # Get the coordinates and size of the region of interest
    roi = img[y:y+h, x:x+w] # Crop the image to the region of interest

    text = pytesseract.image_to_string(roi, lang=lang, config=config) # Use tesseract to extract the text from the region of interest

    df = pd.DataFrame([text], columns=["text"]) # Create a data frame with the text as a column
    return df # Return the data frame

@app.route('/') # Define the route for the home page
def home():
    return render_template('index.html') # Render the index.html template

@app.route('/extract_text', methods=['POST']) # Define the route for extracting text from an image
def extract_text_route():

    if 'image' not in request.files: # Check if there is no image file in the request
        return redirect(request.url) # Redirect to the same page

    file = request.files['image'] # Get the image file from the request

    if not allowed_file(file.filename): # Check if the image file has an allowed extension
        return redirect(request.url) # Redirect to the same page

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED) # Decode the image file from bytes to numpy array
    df = extract_text(img) # Extract text from the image and get a data frame

    output_file = 'output_file.xlsx' # Define the output file name

    # Try to read the existing output file
    try:
        existing_df = pd.read_excel(output_file) # Read the existing data frame from the output file
        new_df = pd.concat([existing_df, df], ignore_index=True) # Concatenate the existing and new data frames
        new_df.to_excel(output_file, index=False) # Save the new data frame to the output file
    # If output file does not exist or cannot be read, create a new one
    except:
        df.to_excel(output_file, index=False) # Save the new data frame to a new output file

    return redirect(url_for('home')) # Redirect to the home page

