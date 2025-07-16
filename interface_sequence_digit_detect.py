import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw

# Load model globally
model = load_model('Trained_Model.h5')

# Prediction pipeline (used by both image upload and canvas draw)
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Could not load image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        messagebox.showinfo("Result", "No digits detected.")
        return

    areas = [cv2.contourArea(c) for c in contours]
    mean_area = np.mean(areas)
    min_pixel_threshold = mean_area /3

    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_pixel_threshold]
    filtered_contours = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[0])

    sequence_digits = []

    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        object_image = gray[y:y + h, x:x + w]

        if np.mean(object_image) > 100:
            object_image = 255 - object_image

        resized = cv2.resize(object_image, (100, 100))
        padded = np.zeros((150, 150), dtype=np.uint8)
        padded[25:125, 25:125] = resized

        dilated = cv2.dilate(padded, np.ones((7, 7), np.uint8), iterations=1)
        blurred = cv2.GaussianBlur(dilated, (9, 9), sigmaX=0)
        processed = cv2.resize(blurred, (28, 28))

        normalized = processed / 255.0
        input_img = np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)

        predictions = model.predict(input_img, verbose=0)
        predicted_digit = np.argmax(predictions)
        confidence = np.max(predictions)

        sequence_digits.append(str(predicted_digit))

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(predicted_digit), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        print(f"Object {i + 1}: Predicted = {predicted_digit}, Confidence = {confidence:.2f}")

    print("Recognized Sequence:", ''.join(sequence_digits))
    messagebox.showinfo("Result", f"Recognized Sequence: {''.join(sequence_digits)}")

    cv2.imshow("Detected Digits", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def upload_and_process():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", ".png;.jpg;.jpeg;.bmp")])
    if file_path:
        process_image(file_path)

# Drawing functionality
canvas_width = 1200
canvas_height = 400
drawn_image_path = "canvas_draw.png"

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill="white")

def save_and_predict():
    img.save(drawn_image_path)
    process_image(drawn_image_path)

# Initialize previous position
prev_x, prev_y = None, None

def paint(event):
    global prev_x, prev_y

    if prev_x is not None and prev_y is not None:
        canvas.create_line(prev_x, prev_y, event.x, event.y,
                           width=8, fill='black', capstyle=tk.ROUND, smooth=True, splinesteps=36)
        draw.line([prev_x, prev_y, event.x, event.y], fill='black', width=8)
    prev_x, prev_y = event.x, event.y

def reset(event):
    global prev_x, prev_y
    prev_x, prev_y = None, None

# GUI Setup
root = tk.Tk()
root.title("Digit Recognition (Draw or Upload)")
root.geometry("500x600")

title_label = tk.Label(root, text="MNIST Digit Recognition", font=("Helvetica", 16))
title_label.pack(pady=10)

upload_button = tk.Button(root, text="Upload Image", command=upload_and_process,
                          font=("Helvetica", 14), bg="blue", fg="white")
upload_button.pack(pady=10)

separator = tk.Label(root, text="OR", font=("Helvetica", 12))
separator.pack()

# Canvas for drawing
canvas_frame = tk.Frame(root)
canvas_frame.pack(pady=10)

canvas = tk.Canvas(canvas_frame, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

img = Image.new("L", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(img)
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", reset)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas,
                         font=("Helvetica", 12), bg="gray", fg="white")
clear_button.grid(row=0, column=0, padx=10)

predict_button = tk.Button(button_frame, text="Predict Drawing", command=save_and_predict,
                           font=("Helvetica", 12), bg="green", fg="white")
predict_button.grid(row=0, column=1, padx=10)

root.mainloop()
