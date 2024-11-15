import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
from colorizers import *  # Ensure this module is correctly installed

# Initialize colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

# Set up the main application window
root = tk.Tk()
root.title("Image Colorization")
root.geometry("800x600")

# Function to load and preprocess the image
def load_img(img_path):
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")  # Ensure the image is in RGB format
    return np.array(img)  # Convert to NumPy array for compatibility

# Function to load and colorize the selected image
def colorize_image():
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not img_path:
        return  # Exit if no file is selected

    try:
        # Load and preprocess image
        img = load_img(img_path)
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
        display_image(img_path, 'Original Image')
        display_image('output_eccv16.png', 'Output (ECCV 16)')
        display_image('output_siggraph17.png', 'Output (SIGGRAPH 17)')

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to display an image in the GUI
def display_image(image_path, title):
    img = Image.open(image_path).resize((256, 256))
    img_tk = ImageTk.PhotoImage(img)
    label = tk.Label(root, image=img_tk, text=title, compound="top", font=("Arial", 12))
    label.image = img_tk  # Keep a reference to avoid garbage collection
    label.pack(pady=10)

# Button to select and colorize an image
btn = tk.Button(root, text="Select Image to Colorize", command=colorize_image, font=("Arial", 14))
btn.pack(pady=20)

# Start the main GUI loop
root.mainloop()
