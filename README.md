
Image Classification Model 

Overview

This project is an image classification system that identifies various fruits and vegetables using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained on a custom dataset and achieves high accuracy in distinguishingbetween different classes of fruits and vegetables. Additionally, the project includes a web application deployed using Streamlit, allowing users to upload images and get real-time predictions.

 Project Structure

- data_train_path: Directory containing training images.
- data_val_path: Directory containing validation images.
- data_test_path: Directory containing test images.
- model.py: Script to define and train the CNN model.
- app.py: Streamlit app for deploying the model as a web application.
- requirements.txt: List of required Python packages.

 Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:

- Input Layer: Rescaling layer to normalize pixel values.
- Conv2D Layer 1: 16 filters, kernel size 3x3, ReLU activation.
- MaxPooling2D Layer 1
- Conv2D Layer 2: 32 filters, kernel size 3x3, ReLU activation.
- MaxPooling2D Layer 2
- Conv2D Layer 3: 64 filters, kernel size 3x3, ReLU activation.
- MaxPooling2D Layer 3
- Flatten Layer
- Dropout Layer: Dropout rate of 0.2.
- Dense Layer 1: 128 units.
- Dense Layer 2: Units equal to the number of classes.

 Training

The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss. The training process includes 20 epochs with a batch size of 32. The dataset is divided into training, validation, and test sets.

 Deployment

The model is deployed as a web application using Streamlit. The app allows users to upload an image, and it displays the predicted class along with the confidence score.

 How to Run

1. Clone the Repository
   ```bash
   git clone https://github.com/yourusername/Fruits-Vegetables-Image-Classification.git
   cd Fruits-Vegetables-Image-Classification
   ```

2. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit App
   ```bash
   streamlit run app.py
   ```

4. Upload an Image
   - Open the Streamlit app in your browser.
   - Upload an image of a fruit or vegetable.
   - View the prediction and confidence score.

 Results

The model achieves an accuracy of 86.07% on test images. The following is an example prediction:

- Input Image: Apple
- Predicted Class: Apple
- Confidence Score: 86.07%

 Future Work

- Increase the dataset size to include more classes.
- Fine-tune the model for higher accuracy.
- Implement additional features in the web app, such as batch predictions.

  ![WhatsApp Image 2024-07-09 at 16 47 25_0a19d973](https://github.com/manepratham120/Image-Classifier/assets/122907546/fdabce9f-fca1-47b4-84ee-cf913aa1a18f)

  ![Screenshot_2024-07-09_164023 1](https://github.com/manepratham120/Image-Classifier/assets/122907546/58b7a568-9598-49a4-9dbf-ce46861c8463)

