# -*- coding: utf-8 -*-
"""
Book: Advanced Deep Learning with Keras (Example, page: 7)
Author: Rowel Atienza

code modified by: jubran akram
"""


# import MNIST dataset
from tensorflow.keras.datasets import mnist
# import NumPy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt

# load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print training and test input sets' shape
print(f"Training Set Shape: {x_train.shape}")
print(f"Test Set Shape: {x_test.shape}")
# number of training and test samples
num_train_samples = x_train.shape[0]
num_test_samples = x_test.shape[0]

# count the number of unique train labels
unique_1, count_1 = np.unique(y_train, return_counts=True)
print(f"{20*'---'} \n Train labels: \n {20*'---'} \n {dict(zip(unique_1, count_1))}")

# count the number of unique test labels
unique_2, count_2 = np.unique(y_test, return_counts=True)
print(f"{20*'---'} \n Test labels: \n {20*'---'} \n {dict(zip(unique_2, count_2))}")

# select "ggplot" style for plotting
plt.style.use('ggplot')
# plot number of test and training sample distribution
fig_1, axs_1 = plt.subplots(1, 2)
axs_1 = axs_1.flatten()
axs_1[0].bar(unique_1, count_1, 0.75, edgecolor='black')
axs_1[0].set_title(f"Total Training Samples: {num_train_samples}")
axs_1[0].set_xlabel('Labels')
axs_1[0].set_ylabel('Count')

axs_1[1].bar(unique_2, count_2, 0.75, edgecolor='black')
axs_1[1].set_title(f"Total Test Samples: {num_test_samples}")
axs_1[1].set_xlabel('Labels')
axs_1[1].set_ylabel('Count')

fig_1.suptitle('Training and Test Data Distribution')


# Extract 36 random samples from the dataset
ind = np.random.randint(0, num_train_samples, size=36)
images = x_train[ind]
labels = y_train[ind]

# plotting these as 5x5 grid
fig, axs = plt.subplots(6, 6)
axs = axs.flatten()
for idx in range(len(axs)):
    axs[idx].imshow(images[idx], cmap='gray')
    axs[idx].axis('off')
fig.suptitle('36 Random Samples from MNIST Training Set')
