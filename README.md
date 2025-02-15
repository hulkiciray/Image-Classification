# Image-Classification

In this repo, we are going to build our image classification model with PyTorch. It will be a basic introductory repo for those who would like a fresh start with computer vision.
Almost all of model training with pytorch paradigm centers around these. If you are using apple silicon like me, you may want to use this line to use your GPU `device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")`

Here are the steps
- **Pytorch Dataset:** In this step we create dataset to have an iterable object. After that we create dataloader so that we have dataset in batches. It helps us feed the model in batches and speeds up the training process.
- **Pytorch Model:** Understanding the pytorch model is all about understanding the shape the data is at each layer, and the main one we need to modify for a task is the final layer.
  - **Here we have 53 targets, so we will modify the last layer for this.**
  ```python
  class Classifier(nn.Module):
    def __init__(self):
        # Here we define all the parts of the model
        super(Classifier, self).__init__()

    def forward(self, x):
        # Connect these parts and return the output
        return output
  ```
- **Pytorch Training Loop:** The order of this is pretty much the same in almost every PyTorch model.
  1. Load in data to the model in batches
  2. Calculate the loss and perform backprop

  ```python
  model = Classifier().to(device)
  for epoch in range(num_epochs):
      # Training phase
      model.train()
      for images, labels in tqdm(train_loader, desc='Training'):
          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      
      # Validation phase
      model.eval()
      running_loss = 0.0
      for images, labels in tqdm(valid_loader, desc='Validation'):
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          loss = criterion(outputs, labels)
  ```
### About the Dataset
This is a very high quality dataset of playing card images. All images are 224 X 224 X 3 in jpg format. All images in the dataset have been cropped so that only the image of a single card is present and the card occupies well over 50% of the pixels in the image. There are 7624 training images, 265 test images and 265 validation images. The train, test and validation directories are partitioned into 53 sub directories , one for each of the 53 types of cards. The dataset also includes a csv file which can be used to load the datasets.
### References
- [Dataset](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification?resource=download)
- [Rob Mulla](https://www.youtube.com/watch?v=tHL5STNJKag)

### Bonus
**glob:**he glob module in Python is used to find all file paths that match a specified pattern, similar to Unix shell-style wildcards.
It’s particularly useful for directory traversal and working with filenames that follow a specific naming pattern.
```python
import glob
files = glob.glob("*.txt") # Find all ".txt" files in the current directory
files = glob.glob("data/*.csv") # Searches for ".csv" files in the "data" directory
files = glob.glob("**/*.py", recursive=True) # search all subdirectories with the ** pattern and recursive=True
```