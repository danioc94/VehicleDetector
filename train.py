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
    print("Loading images finised")
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
    print("length lables: ", len(trainLables))
    #print("Lables: ", trainLables)
    return trainImages, trainLables

#pos_folder = '/home/daniel/Documents/Repositories/VehicleDetector/train/Positive'
pos_folder = '/home/daniel/Documents/CarImages/Positive'
positives = load_images_from_folder(pos_folder)

#neg_folder = '/home/daniel/Documents/Repositories/VehicleDetector/train/Negative'
neg_folder = '/home/daniel/Documents/CarImages/Negative2'
negatives = load_images_from_folder(neg_folder)

# Training images and labels:
trainImages, trainLables = training_data(positives, negatives)
#print("Trainlables: ", trainLables)

# Resizing training image:
resized = resize(trainImages)

# Test images:
test_folder = '/home/daniel/Documents/Repositories/VehicleDetector/test'
test = load_images_from_folder(test_folder)
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
#print("HOG Test: ", hog_test_feats[0])

'''
print("Hog feats: ", hog_feats)
#print("Hog feats type: ", type(hog_feats))
print("Hog feats[0]: ", hog_feats[0])
print("Hog feats[0][0][0]: ", hog_feats[0][0][0])
print("Hog feats[1][0]: ", hog_feats[1][0])
print("Hog feats[0] type: ", type(hog_feats[0]))
'''

# Adjusting data for neural net:
HOG_Data = []
for sample in range(len(hog_feats)):
    HOG_Row = []
    for feature in range(len(hog_feats[sample])):
        HOG_Row.append(hog_feats[sample][feature][0])
        HOG_Data.append(HOG_Row)
#print("HOG_Data: ", HOG_Data)

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
#print("HOG test data: ", HOG_test_Data)

# ANN Predict:
for sample in HOG_test_Data:
    result = ann.predict(np.array([sample], dtype=np.float32))
    print("Result: ", result)

# Saving model:
ann.save("ann_model")
