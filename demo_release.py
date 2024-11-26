import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
from colorizers import *  # Ensure this module is correctly installed

# TensorFlow imports for classification
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Initialize colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

# Load the pre-trained InceptionV3 model
classifier = InceptionV3(weights='imagenet')

# Set up the main application window
root = tk.Tk()
root.title("Image Colorization and Classification")
root.geometry("1200x800")

# Function to load and preprocess the image for colorization
def load_img_for_colorization(img_path):
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")  # Ensure the image is in RGB format
    return np.array(img)  # Convert to NumPy array for compatibility

# Function to classify an image using InceptionV3
def classify_image(img_path):
    try:
        # Load and preprocess the image
        img = load_img(img_path, target_size=(299, 299))  # InceptionV3 requires 299x299 images
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)

        # Perform classification
        predictions = classifier.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=1)  # Get top 1 predictions

        # Format the result
        _, label, _ = decoded_predictions[0][0]  # Extract the first tuple's label
        return label
    except Exception as e:
        return f"Classification error: {e}"

# Function to load and colorize the selected image
def colorize_image():
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not img_path:
        return  # Exit if no file is selected

    try:
        # Load and preprocess image
        img = load_img_for_colorization(img_path)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        tens_l_rs = tens_l_rs.cuda() if torch.cuda.is_available() else tens_l_rs

        # Generate colorizations
        out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
        out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

        # Convert and save output images
        eccv16_img = Image.fromarray((out_img_eccv16 * 255).astype(np.uint8))
        eccv16_img.save('output_eccv16.png')

        siggraph17_img = Image.fromarray((out_img_siggraph17 * 255).astype(np.uint8))
        siggraph17_img.save('output_siggraph17.png')

        # Display images in GUI
        display_images_side_by_side([
            (img_path, 'Original Image'),
            ('output_eccv16.png', 'Output (ECCV 16)'),
            ('output_siggraph17.png', 'Output (SIGGRAPH 17)'),
        ])

        # Perform classification
        class_name = classify_image(img_path)
        messagebox.showinfo("Classification Result", f"Predicted Classes:\n{class_name}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to display images side by side
def display_images_side_by_side(image_data):
    frame = tk.Frame(root)
    frame.pack(pady=20)

    for image_path, title in image_data:
        img = Image.open(image_path).resize((256, 256))
        img_tk = ImageTk.PhotoImage(img)
        label = tk.Label(frame, image=img_tk, text=title, compound="top", font=("Arial", 12))
        label.image = img_tk  # Keep a reference to avoid garbage collection
        label.pack(side="left", padx=10)

# Button to select and process an image
btn = tk.Button(root, text="Select Image to Colorize and Classify", command=colorize_image, font=("Arial", 14))
btn.pack(pady=20)

# Start the main GUI loop
root.mainloop()
