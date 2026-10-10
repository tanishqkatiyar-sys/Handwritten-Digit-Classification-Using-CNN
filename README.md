# MNIST Handwritten Digit Recognition

## 📌 Project Overview

This project is a **Handwritten Digit Recognition System** that predicts handwritten digits from 0 to 9 using Machine Learning and Deep Learning models.

The project uses three different neural network models:

- **Perceptron** – A basic neural network model.
- **ANN (Artificial Neural Network)** – A fully connected neural network.
- **CNN (Convolutional Neural Network)** – A deep learning model that is particularly effective for image recognition.

A **Streamlit web application** allows users to upload handwritten digit images and compare the predictions made by all three models.

## 🎯 Objectives

- Recognize handwritten digits from 0 to 9.
- Train and use Perceptron, ANN, and CNN models.
- Compare predictions from different models.
- Provide an interactive web interface using Streamlit.
- Understand how different neural network architectures perform on image classification.

## 🛠️ Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Pandas
- Pillow (PIL)
- Streamlit
- Jupyter Notebook
- MNIST Dataset

## 📂 Project Structure

```text
MNIST-Handwritten-Digit-Recognition/
│
├── app.py
├── perceptron.keras
├── ann.keras
├── cnn.keras
├── requirements.txt
├── README.md
└── notebook.ipynb
```

**Note:** Replace `notebook.ipynb` with your actual notebook filename. Keep the three trained model files in the same directory as `app.py`, unless your code uses different paths.

## 📊 Dataset

The project uses the **MNIST dataset**, a widely used dataset for handwritten digit recognition.

- It contains 70,000 grayscale images.
- 60,000 images are used for training.
- 10,000 images are used for testing.
- Each image has a size of 28 × 28 pixels.
- Each image represents a digit from 0 to 9.

The images are normalized so that pixel values range from 0 to 1 instead of 0 to 255.

## 🧠 Models Used

### 1. Perceptron

A basic neural network model that learns patterns from input pixels to classify handwritten digits.

### 2. Artificial Neural Network (ANN)

An ANN uses fully connected layers to learn relationships between image pixels and digit classes. The 28 × 28 image is flattened into a one-dimensional array of 784 values.

### 3. Convolutional Neural Network (CNN)

A CNN uses convolutional and pooling layers to learn visual features such as edges, curves, and shapes. It is well suited for image recognition tasks.

## ⚙️ How the Application Works

1. The user uploads an image containing a handwritten digit.
2. The application preprocesses the image.
3. The image is cropped to focus on the handwritten digit.
4. Image preprocessing helps separate the digit from the background, including blue ink where supported.
5. The image is resized to 28 × 28 pixels.
6. Pixel values are normalized.
7. The processed image is passed to the Perceptron, ANN, and CNN models using their required input formats.
8. Each model predicts a digit from 0 to 9.
9. The application displays the predictions so that users can compare the models.

## 🚀 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/tanishqkatiyar-sys/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Replace `YOUR_REPOSITORY_NAME` with the actual repository name.

### Step 2: Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will usually open at:

http://localhost:8501

## 📦 Requirements

Create a `requirements.txt` file containing the dependencies used by your application:

```text
streamlit
tensorflow
numpy
pandas
Pillow
```

For reliable deployment, use package versions compatible with your trained models and deployment Python version.

## ☁️ Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and all three `.keras` model files to your GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using your GitHub account.
4. Click **Create app**.
5. Select your repository and deployment branch.
6. Set the main file path to `app.py`.
7. Click **Deploy**.
8. After successful deployment, Streamlit will provide a public URL for your application.

**Important:** Make sure all model files are available to the deployed app. If the model files are too large for GitHub, use an appropriate model-hosting solution and update the paths in your code.

## 💡 Applications

- Handwritten digit recognition
- Educational demonstrations of neural networks
- Comparison of neural network architectures
- Introduction to image preprocessing and classification
- Deep learning project demonstrations

## 🔮 Future Improvements

- Display prediction confidence scores.
- Allow users to draw digits directly on a canvas.
- Improve recognition of digits photographed under different lighting conditions.
- Evaluate the models using accuracy, precision, recall, and confusion matrices.
- Add more image preprocessing options.

## 👨‍💻 Author

**Tanishq Katiyar**

GitHub: [tanishqkatiyar-sys](https://github.com/tanishqkatiyar-sys)

## 📄 License

This project is intended for educational and learning purposes. Add a license file if you want to specify the terms under which others may use or distribute your code.
