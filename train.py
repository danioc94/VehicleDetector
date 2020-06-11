import cv2
import os
import numpy as np
import random

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def resize(images):
    resized = []
    for im in images:
        i = cv2.resize(im, (64,32))
        resized.append(i)
    return resized

def training_data(positives, negatives):
    trainImages = []
    trainLables = []

    while positives or negatives:
        r = random.random()
        if r <0.5:
            try: 
                trainImages.append(positives.pop())
                trainLables.append([1, 0])
            except:
                pass
        else:
            try:
                trainImages.append(negatives.pop())
                trainLables.append([0, 1])
            except:
                pass
    return trainImages, trainLables

pos_folder = '/home/daniel/Documents/CarImages/Positive100'
positives = load_images_from_folder(pos_folder)
print("Loading positive images finised")

neg_folder = '/home/daniel/Documents/CarImages/Negative100'
negatives = load_images_from_folder(neg_folder)
print("Loading negative images finised")

# Training images and labels:
trainImages, trainLables = training_data(positives, negatives)
#print("Trainlables: ", trainLables)

# Resizing training image:
resized = resize(trainImages)

# Test images:
test_folder = '/home/daniel/Repositories/VehicleDetector/test'
test = load_images_from_folder(test_folder)
print("Loading test images finised")
test_resized = resize(test)

'''
for im in resized:
    cv2.imshow('Positive', im)
    cv2.waitKey()
'''

# HOG parameters:
winSize = (64, 32)
blockSize = (32, 32)  # h x w in cells
blockStride = (32, 32)
cellSize = (16, 16)  # h x w in pixels
nbins = 9  # number of orientation bins

# HOG computation for training images:
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
hog_feats = []
for im in range(len(resized)):
    im_feature = hog.compute(resized[im])
    hog_feats.append(im_feature)

# Compute HOG for test image:
hog_test_feats = []
for im in range(len(test_resized)):
    im_feature = hog.compute(test_resized[im])
    hog_test_feats.append(im_feature)

# Adjusting data for neural net:
HOG_Data = []
for sample in range(len(hog_feats)):
    HOG_Row = []
    for feature in range(len(hog_feats[sample])):
        HOG_Row.append(hog_feats[sample][feature][0])
    HOG_Data.append(HOG_Row)

# ANN Setup:
featureLength = len(hog_feats[0])
ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([featureLength, 64, 32, 2], dtype=np.uint8))
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)

# ANN Training:
for sample in range(len(trainLables)):
    ann.train(np.array([HOG_Data[sample]], dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array([trainLables[sample]], dtype=np.float32))

# Adjusting test data for neral net:
HOG_test_Data = []
for sample in range(len(hog_test_feats)):
    HOG_Row = []
    for feature in range(len(hog_test_feats[sample])):
        HOG_Row.append(hog_test_feats[sample][feature][0])
    HOG_test_Data.append(HOG_Row)

# ANN Predict:
for sample in HOG_test_Data:
    result = ann.predict(np.array([sample], dtype=np.float32))
    print("Result: ", result)

# Save model:
ann.save("ann_model")
