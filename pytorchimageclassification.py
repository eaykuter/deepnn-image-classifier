# -*- coding: utf-8 -*-
"""PyTorchImageClassification.ipynb"""

import kagglehub
gpiosenka_cards_image_datasetclassification_path = kagglehub.dataset_download('gpiosenka/cards-image-datasetclassification')

print('Data source import complete.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm
from PIL import Image
from glob import glob

class CardDataSet(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data = ImageFolder(data_dir, transform=transform)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

  @property
  def classes(self):
    return self.data.classes

"""Creating a test dataset:"""

dataset = CardDataSet(data_dir="/kaggle/input/cards-image-datasetclassification/train")

len(dataset)

image, label = dataset[6]
print(label)
image

data_dir = '/kaggle/input/cards-image-datasetclassification/train'

#this creates a dictionary with folder names as keys and card data as values
#52 classes since we're using playing cards
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
print(target_to_class)

#like a script to resize data
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

#here the script is passed into our data and transformed
data_dir = "/kaggle/input/cards-image-datasetclassification/train"
dataset = CardDataSet(data_dir=data_dir, transform=transform)

#checking if the transformation worked (should be 3 colors, and  128 by 128 pixels)
image, label = dataset[100]
image.shape

#splitting the data into batches for easier/faster processing
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#checking the first batch of pictures (32 pics, 3 color, and 128x128 dimension)
for images, labels in dataloader:
    break

print(images.shape, labels.shape)
print(labels) #labels in this batch

"""Here, we are determining how our model is going to behave and how many classes it will be able to specify.

53 classes set as default because there are 53 card types to distinguish between
"""

class CardClassifier(nn.Module):
  def __init__(self, num_classes=53):
    super(CardClassifier, self).__init__()

    self.base_model = timm.create_model("efficientnet_b0", pretrained=True)

    #cutting the last later of the network, so that we can specify output class number
    # the * takes the items out of the list and passes them as is
    self.features = nn.Sequential(*list(self.base_model.children())[:-1])

    #hidden layer feature numbers (efficientnet_b0 default)
    enet_out_size = 1280

    #mapping the 1280 features into only 53 for our card dataset
    self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(enet_out_size, num_classes))

  def forward(self, x):
    x = self.features(x)
    output = self.classifier(x)

    return output

"""Testing if it works:"""

num_classes = len(target_to_class)
model = CardClassifier(num_classes=num_classes)

example_out = model(images)
example_out.shape #32 photos, 53 classes

"""Training the model:"""

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(criterion(example_out, labels))
print(example_out.shape, labels.shape)

"""Setting up the data and splitting it into training and validation for training and test for final evaluation"""

train_folder = "/kaggle/input/cards-image-datasetclassification/train/"
valid_folder = "/kaggle/input/cards-image-datasetclassification/valid/"
test_folder = "/kaggle/input/cards-image-datasetclassification/test/"

#using ImageFolder here for modularity, but CardDataSet is the default
train_dataset = ImageFolder(train_folder, transform=transform)
val_dataset = ImageFolder(valid_folder, transform=transform)
test_dataset = ImageFolder(test_folder, transform=transform)

#shuffling is not needed for validation and test datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

"""Now the training loop for the neural network"""

num_epochs = 5
training_losses, val_losses = [], []

#this is for accelerating the training with nvidia's cuda architecture
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    #looping over the batches of data
    for images, labels in tqdm(train_loader, desc="Training loop"):
        # Move inputs and labels to the device as well
        images, labels = images.to(device), labels.to(device)

        #clearing out the optimizer
        optimizer.zero_grad()

        outputs = model(images)

        # the loss between the guess and the labels
        loss = criterion(outputs, labels)

        #back propogation
        loss.backward()

        #updating the weights
        optimizer.step()

        #calculates the loss on this run weighted by the batch size
        running_loss += loss.item() * labels.size(0)

    # after running on each batch, calculates the average loss on each epoch
    training_loss = running_loss / len(train_loader.dataset)
    training_losses.append(training_loss)

    model.eval()
    running_loss = 0.0

    #deactivates gradient calculation for faster calculations,
    #since we don't need it for val
    with torch.no_grad():

      # same as training loop, but without back propogation
      for images, labels in tqdm(val_loader, desc="Validation loop"):


          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          loss = criterion(outputs, labels)
          running_loss += loss.item() * labels.size(0)

      val_loss = running_loss / len(val_loader.dataset)
      val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {training_loss}, Validation loss: {val_loss}")

"""Graphing the losses over time.

If only the training goes down while the validation is going up, it means we are overfitting the training data.
"""

plt.plot(training_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()

"""Time to play around!"""

# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
  model.eval()

  with torch.no_grad():
    image_tensor = image_tensor.to(device)

    outputs = model(image_tensor)

    probabilities = nn.functional.softmax(outputs, dim=1)

  return probabilities.cpu().numpy().flatten()

def visualize_predictions(original_image, probabilities, class_names):
  fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

  axarr[0].imshow(original_image)
  axarr[0].axis("off")

  # Display predictions
  axarr[1].barh(class_names, probabilities)
  axarr[1].set_xlabel("Probability")
  axarr[1].set_title("Class Predictions")
  axarr[1].set_xlim(0, 1)

  plt.tight_layout()
  plt.show()

"""Example Usage:"""

test_image = "/kaggle/input/cards-image-datasetclassification/test/five of diamonds/5.jpg"
original_image, image_tensor = preprocess_image(test_image, transform)
probabilities = predict(model, image_tensor, device)

class_names = dataset.classes
visualize_predictions(original_image, probabilities, class_names)

test_images = glob('/kaggle/input/cards-image-datasetclassification/test/*/*')
test_examples = np.random.choice(test_images, 10)

for example in test_examples:
  original_image, image_tensor = preprocess_image(example, transform)
  probabilities = predict(model, image_tensor, device)

  class_names = dataset.classes
  visualize_predictions(original_image, probabilities, class_names)

  predicted_idx = np.argmax(probabilities)
  print(f"Predicted: {class_names[predicted_idx]}")
  print()