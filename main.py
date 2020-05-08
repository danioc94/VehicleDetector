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
for im in resized:
    cv2.imshow('Positive', im)
    cv2.waitKey()