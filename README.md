Crop Covering Mobile App
Overview

Crop Covering Mobile App is a deep learning project that segments satellite images to map crop coverage. Using U-Net and a Masked Attention Transformer with Dense Upsampling Convolutions (DUC), the system identifies crop types (pomegranate, date palm, and background) from high-resolution TIFF images.

The trained models are exported to TensorFlow Lite, enabling deployment in a mobile app where users can upload or select an image and see color-coded segmentation masks along with coverage percentages.

Features

Crop segmentation into background, pomegranate, and date palm.

Coverage calculation to estimate percentage area of each crop.

Multiple model variants: U-Net and Transformer + DUC.

TensorFlow Lite export for mobile deployment.

Visualization with segmentation overlays.

Tech Stack

Deep Learning: TensorFlow, Keras

Computer Vision: OpenCV, Matplotlib

Data: High-resolution satellite TIFF imagery

Deployment: TensorFlow Lite, React Native (starter app included)


How It Works

Train models using scripts in training/.

Evaluate and visualize predictions with testing/.

Run inference on new data with inference/.

Export trained models to TensorFlow Lite with export/export_to_tflite.py.

Load the .tflite model inside the React Native app for mobile deployment.

Example Output

Input: Satellite image tile.

Output: Segmentation mask overlay (Green = Pomegranate, Blue = Date Palm, Black = Background) with coverage percentages.

Future Work

Extend to more crop types using larger datasets.

Integrate geolocation for region-based queries.

Add crop health monitoring features such as NDVI-based indices.
