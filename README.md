# Plant Health Detector

## Overview

The **Plant Health Detector** is a machine learning-based web application that identifies diseases in tomato and potato leaves using a Convolutional Neural Network (CNN). Built with TensorFlow and deployed via Streamlit, it allows users to upload leaf images and receive disease predictions with a user-friendly interface. This project showcases expertise in deep learning, image processing, and web application development, as part of my portfolio at [Firoz Khan](https://github.com/firozzorif).

## Features

* **Disease Detection**: Classifies diseases in tomato and potato leaves, including Early Blight, Late Blight, Bacterial Spot, and more, with a "healthy" class for unaffected leaves.
* **CNN Model**: Utilizes a TensorFlow/Keras-based CNN with multiple convolutional and pooling layers for accurate image classification.
* **Data Augmentation**: Applies transformations (rotation, zoom, flip) to enhance model robustness using `ImageDataGenerator`.
* **Streamlit Interface**: Provides an intuitive web interface for uploading multiple images (minimum 3) and displaying predictions with visual feedback.
* **Model Training**: Includes training and fine-tuning scripts with early stopping and model checkpointing to save the best-performing model.
* **Visualization**: Plots training/validation accuracy and loss to evaluate model performance.

## Technologies Used

* **Languages**: Python
* **Libraries**: TensorFlow, Keras, Streamlit, NumPy, Matplotlib, Pyngrok
* **Tools**: Google Colab, Google Drive, Ngrok
* **Concepts**: Deep Learning, Convolutional Neural Networks, Image Processing, Web Application Deployment

## Prerequisites

* Google Colab account with access to a **T4 GPU** runtime (recommended for faster training).
* Google Drive account to store the dataset and model files.
* Ngrok account with an authentication token for Streamlit deployment.
* Dataset: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (or equivalent).

## Setup Instructions

1. **Open in Google Colab**:

   * Copy the provided notebook code into a new Google Colab notebook.
   * **Select T4 GPU Runtime**:

     * Go to `Runtime` > `Change runtime type`.
     * Choose `T4 GPU` under Hardware Accelerator.

2. **Mount Google Drive**:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Prepare the Dataset**:

   * Upload `New Plant Diseases Dataset.zip` to your Google Drive.
   * Update the `zip_file_path` in the code to your dataset path.
   * Ensure it contains `train` and `valid` folders with disease-labeled subfolders.

4. **Install Dependencies**:

   ```bash
   !pip install pyngrok streamlit
   ```

5. **Set Up Ngrok**:

   ```python
   from pyngrok import ngrok
   ngrok.set_auth_token('YOUR_NGROK_AUTH_TOKEN')
   ```

6. **Run the Notebook**:

   * Unzip the dataset.
   * Train the CNN model (start with 10 epochs, fine-tune up to 50 with early stopping).
   * Save the model: `Final_Updated_Plant_Disease_Model_latest.h5`.
   * Plot training/validation accuracy and loss.
   * Predict sample test images.

7. **Deploy the Streamlit App**:

   * Launch `app.py` with Streamlit.
   * Use the Ngrok URL (e.g., `http://<random>.ngrok.io`) to access the app.
   * Upload 3 or more images for classification.

## Dataset Structure

```
/content/New_Plant_Diseases_Dataset/
├── train/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   ├── Potato___healthy/
│   ├── Tomato___Bacterial_spot/
│   ├── ...
└── valid/
    ├── Potato___Early_blight/
    ├── Potato___Late_blight/
    ├── ...
```

## Usage

### Training the Model

* Run all training cells.
* Save the trained model to Google Drive.
* Visualize training with accuracy and loss plots.

### Testing Predictions

* Update `test_image_paths` with new image paths.
* Run the prediction function to see classification results.

### Streamlit App

* Access the Ngrok URL.
* Upload at least 3 leaf images.
* The app displays:

  * Uploaded image
  * Plant type (tomato/potato)
  * Disease or healthy status

## Troubleshooting

* **T4 GPU Not Available**: Use CPU but expect slower performance.
* **Path Errors**: Ensure correct paths for dataset and model.
* **Ngrok Errors**: Recheck Ngrok token and port configuration.
* **Mismatched Classes**: Ensure the `train` and `valid` folders have the same class structure.

## Future Enhancements

* Add support for more crops.
* Apply advanced preprocessing (e.g., histogram equalization).
* Deploy on **Streamlit Cloud** for permanent hosting.
* Show prediction confidence scores.

## Repository

* **GitHub**: [Plant Health Detector](https://github.com/firozzorif/Plant-Health-Detector) *(Create the repo if not present)*
* **Status**: Actively maintained

## Demo

* **Planned**: Streamlit Cloud deployment.
* **Temporary Access**: Via Ngrok URL after notebook execution.

## Notes

* **GPU Training**: T4 GPU is recommended for faster results.
* **Dataset Prep**: Crucial to structure the dataset correctly.
* **Deployment**: Ngrok is temporary—use Streamlit Cloud for a stable link.
* **Portfolio**: This is part of my machine learning portfolio. See more at [firozzorif](https://github.com/firozzorif).

## Contact

* **Name**: Firoz Khan
* **Email**: [firozzorif2003@gmail.com](mailto:firozzorif2003@gmail.com)
* **GitHub**: [firozzorif](https://github.com/firozzorif)

