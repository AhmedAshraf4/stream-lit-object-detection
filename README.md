
# Streamlit Object Detection

## Description
This project demonstrates object detection using Streamlit with a YOLO model from the Ultralytics library. Users can upload an image, select a YOLO model variant, and analyze detected objects with their counts displayed.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AhmedAshraf4/stream-lit-object-detection.git
   cd stream-lit-object-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the 5 [YOLOv8 models](https://docs.ultralytics.com/tasks/detect/#models) and put them in the same directory
   Ensure you have Python installed along with necessary packages like Streamlit, OpenCV, Pillow, and Ultralytics.

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   
2. Open your web browser and go to `localhost:8501`.

3. Upload an image (JPEG, PNG) and choose a YOLO model variant.

4. Click on "Analyse image" to detect objects and view results.

## Credits
- Developed as part of the Slash Internship.
- [Task link](https://drive.google.com/file/d/1WP6ePJ-tfqQ_QXra1JBoF_pcD9s35kXO/view?usp=sharing)

