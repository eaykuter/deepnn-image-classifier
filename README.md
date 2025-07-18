# PyTorch Playing Card Classifier 🃏

A deep learning project that uses Python and the PyTorch framework to build a highly accurate image classifier. This model leverages **transfer learning** with a pre-trained EfficientNet-B0 architecture to classify a dataset of 53 unique playing cards.

![Prediction Example]([https://imgur.com/a/LliFxb3.png](https://imgur.com/a/LliFxb3))

---

## Key Technologies & Concepts

* **Framework**: PyTorch
* **Core Technique**: Transfer Learning
* **Model Architecture**: EfficientNet-B0 (via the `timm` library)
* **Tools**: Kaggle API, NumPy, Matplotlib
* **Dataset**: [Cards Image Dataset Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)

---

## 🏛️ Model Architecture: Transfer Learning

Instead of training a neural network from scratch, this project uses **transfer learning**. This technique takes a powerful, pre-trained model and adapts it for a new, specific task.

1.  **Base Model**: We start with **EfficientNet-B0**, a state-of-the-art model pre-trained on the massive ImageNet dataset. This model already knows how to recognize a rich set of features like edges, textures, and shapes.
2.  **Feature Extraction**: The core layers of EfficientNet-B0 are used as a fixed feature extractor. We pass our card images through these layers to get a high-level numerical representation of each image.
3.  **Custom Classifier**: The original final layer of EfficientNet (which classifies 1000 ImageNet classes) is removed. It's replaced with a new, custom `nn.Linear` layer that takes the features from the base model and maps them to our **53 playing card classes**.
4.  **Fine-Tuning**: Only the new, custom classifier layer is trained on our card dataset. This process is much faster and requires less data than training an entire network from the ground up.

This approach allows us to achieve high accuracy without the need for a massive dataset or extensive computation time.

---

## ⚙️ Setup and Installation

To run this project, you'll need Python 3 and the required packages. It's recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/eaykuter/deepnn-image-classifier.git]
    cd [deepnn-image-classifier]
    ```

2.  **Set up Kaggle API:**
    This project downloads its dataset directly from Kaggle. You will need to [create a Kaggle API token](https://www.kaggle.com/docs/api) (`kaggle.json`) and place it in the appropriate directory (e.g., `~/.kaggle/` on Linux/macOS).

3.  **Create `requirements.txt`:**
    Create a file named `requirements.txt` with the following contents:
    ```txt
    torch
    torchvision
    timm
    kaggle
    pandas
    numpy
    matplotlib
    pillow
    tqdm
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

The entire workflow—from data loading to training and prediction—is contained within the `PyTorchImageClassification.ipynb` notebook.

1.  **Open and run the notebook** in an environment like Jupyter Lab, Jupyter Notebook, or VS Code.
2.  The notebook will first **download the dataset** using the Kaggle API.
3.  You can **run the cells sequentially** to perform data preprocessing, model definition, training, and validation.
4.  The final cells of the notebook provide functions to **test the trained model** on individual images from the test set and visualize the prediction probabilities.

---

## 📊 Results

The model was trained for 5 epochs, with performance monitored on a validation set to prevent overfitting. The loss curves below show that the model learned effectively, with both training and validation loss decreasing consistently.
![Validation Loss]([https://imgur.com/a/eZqnO9T.png](https://imgur.com/a/eZqnO9T))
