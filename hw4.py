# Adam Craig
# Deep Learning
# HW4
# Unsupervised learning

from numpy import array
from torch import Tensor
import torch


import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt

K_MEANS_SIZE = 100
K_MEANS_FEATURE_SIZE = 1000
EPOCHS = 20

# PART 1
# K-means clustering on MNIST
class N_Dataset(Dataset):
    def __init__(self, N: int, to_np: bool, root, train, download, transform):
        self.dataset = datasets.MNIST(
            root=root, train=train, download=download, transform=transform
        )
        self._len = N*10
        self.counts = { i : 0 for i in range(10)}
        self.sample_counts = { i : None for i in range(10)}
        self.imgs = list()
        self.labels = list()
        self.sample_pairs = list()
        self._populate(N, to_np)

    def _populate(self, N: int, to_np: bool):
        # populate the N-subset that will be used
        # create random permutation each time to vary experiments
        # f determines whether or not subarrays are numpy arrays
        if to_np:
            f = lambda x : x.numpy()
        else:
            f = lambda x : x 
    
        # pi = np.random.permutation(self.dataset.__len__())
        pi = lambda x : x 
        for idx in range(self.dataset.__len__()):
            X, label = self.dataset.__getitem__(pi(idx))
            if self.counts[label] < N:
                if self.sample_counts[label] is None:
                    self.sample_counts[label] = f(X)
                self.imgs.append(f(X))
                self.labels.append(label)
                self.counts[label] += 1
        # create list
        for i in range(10):
            self.sample_pairs.append(self.sample_counts[i])

    def __len__(self):
        return self._len 

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx], self.imgs[idx]

training_dataset = N_Dataset(K_MEANS_SIZE, True, root="data", train=True, download=True, transform=transforms.ToTensor())
# X = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=0)
X = np.array(training_dataset.imgs)
X.squeeze()
X = np.reshape(X, (K_MEANS_SIZE*10, 28*28))

init_sample = np.array(training_dataset.sample_pairs)
init_sample.squeeze()
init_sample = np.reshape(init_sample, (10, 28*28))

# set init to give centroids
# add random sampling array to N_dataset
kmeans = KMeans(n_clusters=10, init=init_sample).fit(X)
Y_clusters = kmeans.labels_

# Now create confusion matrix
confusion_matrix = np.zeros([10, 10])

for idx in range(len(Y_clusters)):
    X, label, _ = training_dataset[idx]
    i, j = label, Y_clusters[idx]
    confusion_matrix[i, j] += 1

accuracy = 0
for idx in range(10):
    accuracy += confusion_matrix[idx, idx]

print("PART ONE")
print(f"Accuracy: {accuracy/K_MEANS_SIZE*10}")
print("Confusion matrix:")
print(confusion_matrix)

# PART 2
# k-means clustering using feature vectors
training_dataset = N_Dataset(K_MEANS_FEATURE_SIZE, False, root="data", train=True, download=True, transform=transforms.ToTensor())

# autoencoder and training loop from 
# https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(model, training_dataset, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, 
                                 weight_decay=1e-5) # <--
    train_loader = torch.utils.data.DataLoader(training_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _, true_value = data
            recon = model(img)
            if torch.equal(true_value, img):
                loss = criterion(recon, true_value)
            else:
                loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs

model = Autoencoder()
outputs = train(model, training_dataset, num_epochs=EPOCHS)

# construct feature vectors
samples = list()
for idx in range(len(training_dataset.sample_pairs)):
    samples.append(model.encoder(torch.unsqueeze(training_dataset.sample_pairs[idx],0)).detach().numpy())
samples = np.array(samples)
samples = np.squeeze(samples)

feature_list = list()
for idx in range(len(training_dataset.imgs)):
    feature_list.append(model.encoder(torch.unsqueeze(training_dataset.imgs[idx],0)).detach().numpy())
feature_list = np.array(feature_list)
feature_list = np.squeeze(feature_list)

# set init to give centroids
# add random sampling array to N_dataset
kmeans = KMeans(n_clusters=10, init=samples).fit(feature_list)
Y_clusters = kmeans.labels_

# Now create confusion matrix
confusion_matrix = np.zeros([10, 10])

for idx in range(len(Y_clusters)):
    X, label, _ = training_dataset[idx]
    i, j = label, Y_clusters[idx]
    confusion_matrix[i, j] += 1

accuracy = 0
for idx in range(10):
    accuracy += confusion_matrix[idx, idx]

print("PART 2A")
print(f"Accuracy: {accuracy/K_MEANS_FEATURE_SIZE*10}")
print("Confusion matrix:")
print(confusion_matrix)

# Now perform PCA on the feature vectors
pca = KernelPCA(n_components=10)
X = pca.fit_transform(feature_list)
Y = pca.transform(samples)

# set init to give centroids
# add random sampling array to N_dataset
kmeans = KMeans(n_clusters=10, init=Y).fit(X)
Y_clusters = kmeans.labels_

# Now create confusion matrix
confusion_matrix = np.zeros([10, 10])

for idx in range(len(Y_clusters)):
    X, label, _ = training_dataset[idx]
    i, j = label, Y_clusters[idx]
    confusion_matrix[i, j] += 1

accuracy = 0
for idx in range(10):
    accuracy += confusion_matrix[idx, idx]

print("PART 2B")
print(f"Accuracy: {accuracy/K_MEANS_FEATURE_SIZE*10}")
print("Confusion matrix:")
print(confusion_matrix)

# PART 3

# create a funciton to add salt and pepper noise to
# the mnist images
def add_noise(img):
    # make sure we have a greyscale MNIST image
    assert img.shape == (1, 28, 28)
    new_img = torch.tensor(img)
    # add 20 random white pixels
    for _ in range(20):
        x_coord = np.random.random_integers(low=0, high=28-1)
        y_coord = np.random.random_integers(low=0, high=28-1)
        new_img[0,x_coord,y_coord] = 1
    # add 20 random black pixels
    for _ in range(20):
        x_coord = np.random.random_integers(low=0, high=28-1)
        y_coord = np.random.random_integers(low=0, high=28-1)
        new_img[0,x_coord,y_coord] = 0
    return new_img

class NoiseDataset(Dataset):
    def __init__(self, root, train, download, transform):
        self.dataset = datasets.MNIST(root=root, train=train, download=download, transform=transform)
        self._array = list()
        self.samples = { i : None for i in range(10)}
        self._populate(train)
    
    def _populate(self, train):
        if train:
            for idx in range(self.dataset.__len__()):
                X, label = self.dataset[idx]
                Y = add_noise(X)
                self._array.append((Y, label, X))
                if self.samples[label] == None:
                    self.samples[label] = (Y, label, X)
        else:
            for idx in range(self.dataset.__len__()):
                X, label = self.dataset[idx]
                self._array.append((X, label, X))

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self._array[idx]

training_dataset = NoiseDataset(root="data", train=True, download=True, transform=transforms.ToTensor())

# create and train the autoencoder
model = Autoencoder()
outputs = train(model, training_dataset, num_epochs=EPOCHS, learning_rate=1e-3)

# collect 10 samples from each class
display_images = list()
for idx in range(10):
    Y, label, X = training_dataset.samples[idx]
    output = model.forward(torch.unsqueeze(Y,0)).detach().numpy()
    display_images.append(X)
    display_images.append(Y)
    display_images.append(output)
display_images = np.array(display_images)

# taken from https://stackoverflow.com/questions/41432568/show-multiple-image-in-matplotlib-from-numpy-array
fig = plt.figure(figsize=(4, 12))  # width, height in inches

for i in range(30):
    sub = fig.add_subplot(10, 3, i + 1)
    sub.imshow(display_images[i].squeeze(), interpolation='nearest')

print("PART THREE")
plt.show()
