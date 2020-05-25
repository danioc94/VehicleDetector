import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print(filename)
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
    trainImages = positives + negatives
    trainLables = []
    for i in range(len(trainImages)):
        if i < 10:
            trainLables.append(1)
        else:
            trainLables.append(-1)
    return trainImages, trainLables

pos_folder = '/home/daniel/Documents/Repositories/VehicleDetector/train/Positive'
positives = load_images_from_folder(pos_folder)

neg_folder = '/home/daniel/Documents/Repositories/VehicleDetector/train/Negative'
negatives = load_images_from_folder(neg_folder)

# Training images and labels:
trainImages, trainLables = training_data(positives, negatives)
print("Trainlables: ", trainLables)

# View positives:
'''
for im in positives:
    cv2.imshow('Positive', im)
    cv2.waitKey()
#print(images)
'''

# View resized:
resized = resize(trainImages)
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

# HOG computation
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
hog_feats = []

for im in range(len(resized)):
    im_feature = hog.compute(resized[im])
    hog_feats.append(im_feature)

'''
print("Hog feats: ", hog_feats)
#print("Hog feats type: ", type(hog_feats))
print("Hog feats[0]: ", hog_feats[0])
print("Hog feats[0][0][0]: ", hog_feats[0][0][0])
print("Hog feats[1][0]: ", hog_feats[1][0])
print("Hog feats[0] type: ", type(hog_feats[0]))
print("Feature length: ", featureLength)
'''

featureLength = len(hog_feats[0])
example = []
for i in range(len(hog_feats[0])):
    example.append(hog_feats[0][i][0])
#print("example: ", example)

# ANN Setup:
ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([featureLength, 64, 32, 2], dtype=np.uint8))
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)

numpyArray = np.array([[1, 1, 1, 1, 0, 1, 1, 0, 0, 1]], dtype=np.float32)
#print("numpyArray type: ", type(numpyArray))

# ANN Training:
ann.train(np.array([example], dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array([[1, 0]], dtype=np.float32))

# ANN Predict:
#result = ann.predict(np.array([[1.4, 1.5, 1.2, 2., 2.5, 2.8, 3., 3.1, 3.8]], dtype=np.float32))
#print(result)