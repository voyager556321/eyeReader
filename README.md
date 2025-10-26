# Eye Reader PDF

Eye Reader PDF is a Python application that allows you to read PDF books using eye tracking. By using your webcam, the program detects eye movement and scrolls the PDF vertically based on where your eyes are looking. You can also pause scrolling by closing your eyes.

## Features

- Eye-tracking-based PDF scrolling using your webcam.
- Face and iris detection using MediaPipe Face Mesh.
- Eye closure detection to pause scrolling.
- Overlay of a small webcam view in the top-left corner of the PDF window.
- Works with multi-page PDFs, automatically stacking all pages vertically.
- Adjustable scroll speed.

## Requirements

- Python 3.8+
- OpenCV (cv2)
- MediaPipe (mediapipe)
- PyMuPDF (fitz)
- Pillow (PIL)
- NumPy (numpy)

Install the dependencies using pip:

``bash
pip install opencv-python mediapipe PyMuPDF Pillow numpy


## Usage

Run the script from the command line with a PDF file as an argument:

``bash
python eyereader_pdf.py your_book.pdf



- Escape (Esc) key: Exit the program.

## How It Works

1. **PDF Loading**  
   - The script opens the PDF using PyMuPDF and converts each page into an image.
   - All pages are stacked vertically into one large image.

2. **Eye Tracking**  
   - The webcam feed is processed using MediaPipe Face Mesh to detect face landmarks and irises.
   - The vertical position of the irises relative to a neutral position determines scrolling.

3. **Pause Functionality**  
   - If eyes are closed, scrolling is paused automatically.

4. **Display**  
   - The program shows a window of size 800×600 pixels showing the relevant portion of the PDF.
   - The webcam feed is overlaid in the top-left corner for reference.

## Configuration

- `scroll_multiplier`: Adjust this value to change the scroll sensitivity. Default is 250.
- `window_w` and `window_h`: Dimensions of the display window. Default is 800×600.

