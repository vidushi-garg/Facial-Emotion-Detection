#This python file calculates the mean and standard deviation of the dataset

import os
from PIL import Image
import numpy as np
# from scipy.misc import imread
from matplotlib.pyplot import imread

filepath = '../cleaned_dataset/'
pathDir=os.listdir(filepath)

# mean
total = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = imread(os.path.join(filepath, filename))
    total = total + np.sum(img[:, :])


num = len(pathDir)*48*48
mean = (total / num)
print("mean"+str(mean))

# std
total = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = imread(os.path.join(filepath, filename))
    total = total + np.sum((img[:, :] - mean) ** 2)

std = np.sqrt(total / num)
print("std"+str(std))

