import cv2
import os

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

pos_folder = '/home/daniel/Documents/Repositories/VehicleDetector/train/Positive'
positives = load_images_from_folder(pos_folder)

# View positives:
'''
for im in positives:
    cv2.imshow('Positive', im)
    cv2.waitKey()
#print(images)
'''

# View resized:
resized = resize(positives)
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
#hog_feats = hog.compute(resized[0])
#print(hog_feats)

hog_feats = []
print("Length resized: ", len(resized))


for im in range(len(resized)):
    im_feature = hog.compute(resized[im])
    hog_feats.append(im_feature)

print("Hog feats: ", hog_feats)
print("Hog feats type: ", type(hog_feats))
print("Hog feats[0]: ", hog_feats[0][0][0])
print("Hog feats[1]: ", hog_feats[1][0])
print("Hog feats[0] type: ", type(hog_feats[0]))
