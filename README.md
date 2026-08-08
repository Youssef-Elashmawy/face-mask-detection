Face Mask Detection
A deep learning model that classifies whether a person in an image is wearing a face mask, built with transfer learning (MobileNetV2) on TensorFlow/Keras.
Overview
This project follows a full data preprocessing and model development pipeline: loading and exploring the dataset, cleaning it (duplicates, missing/corrupted files, outliers), visualizing distributions, checking class balance, training a CNN with transfer learning, and evaluating it with standard classification metrics.
Dataset
Face Mask Detection Dataset by Omkar Gurav on Kaggle — 7,553 images across two classes: `with_mask` and `without_mask`.
Pipeline
Load & Explore — loaded 7,553 images into a filepath/label table; checked shapes, dtypes, and formats.
Duplicates — detected via perceptual hashing (`imagehash`); removed 657 exact duplicates (8.7% of the dataset).
Missing Values — checked all images for corruption; 0 unreadable files found.
Outliers — flagged resolution, aspect ratio, and brightness outliers via IQR; removed 3 extreme-brightness images. Resolution outliers were handled by standard resizing in the data pipeline rather than removal.
Visualizations — 7 plots covering class balance, brightness/aspect ratio distributions, width vs. height by class, feature correlation, and a multivariate pairplot.
Class Balance — dataset found to be near-balanced (49.2% / 50.8%); class weights computed as a safeguard.
Model Training — MobileNetV2 (frozen ImageNet base) + custom classification head, trained with data augmentation and early stopping. Final result: 99.11% train accuracy, 98.45% validation accuracy (0.66% gap — good fit, no overfitting).
Evaluation — measured on a held-out test set never seen during training.
Results
Metric	Score
Accuracy	98.07%
Precision	98.84%
Recall	97.33%
F1 Score	98.08%
ROC-AUC	99.89%
Compared to a majority-class baseline (50.77% accuracy), the model improves performance by 47.29 percentage points.
Files
`face_mask_detection.ipynb` — full notebook: preprocessing, training, and evaluation
`face_mask_detector.h5` — trained model weights
How to Run
Download the dataset from Kaggle
Install dependencies:
```
   pip install -r requirements.txt
   ```
Open `face_mask_detection.ipynb` and run the cells in order (update the dataset path to match your environment)
Using the Trained Model
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model("face_mask_detector.h5")

img = load_img("path/to/image.jpg", target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
label = "without_mask" if prediction > 0.5 else "with_mask"
print(label)
```
Tech Stack
TensorFlow / Keras (MobileNetV2 transfer learning)
OpenCV, Pillow
scikit-learn
pandas, NumPy
Matplotlib, Seaborn
imagehash
