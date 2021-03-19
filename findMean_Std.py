import os
from PIL import Image
import numpy as np
# from scipy.misc import imread
from matplotlib.pyplot import imread

filepath = '../dataset/images'
pathDir=os.listdir(filepath)

# mean
channel = 0

total = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    print(idx)
    img = imread(os.path.join(filepath, filename))

    # total_pixel = total_pixel + img.shape[0] * img.shape[1]

    total = total + np.sum(img[:, :])
num = len(pathDir)*48*48
mean = (total / num)
print("mean"+str(mean))
# G_std = sqrt(G_total / total_count)
# B_std = sqrt(B_total / total_count)

# std
channel = 0
total = 0
# total_pixel = 0
for idx in range(len(pathDir)):
    print(idx)
    filename = pathDir[idx]
    img = imread(os.path.join(filepath, filename))

    # total_pixel = total_pixel + img.shape[0] * img.shape[1]

    total = total + np.sum((img[:, :] - mean) ** 2)

std = np.sqrt(total / num)
# G_std = sqrt(G_total / total_count)
# B_std = sqrt(B_total / total_count)
print("mean"+str(mean))
print("std"+str(std))

