import tensorflow as tf
import os
import cv2
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
tf.config.run_functions_eagerly(True)
from PIL import Image


# The Convolutional Base
def create_model():
    # The Convolutional Base
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Now the dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  run_eagerly=True)
    return model

def add_salt_pepper_noise(X_imgs, img_ids, train_label_dictionary):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

    labels = []
    i= 0
    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
        labels.append(train_label_dictionary[img_ids[i]])
        i=i+1
    
    return X_imgs_copy, labels

def add_color_augmentation(X_imgs, img_ids, train_label_dictionary):
    print('adding color augmentation..')

    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape

    i=0
    labels = []
    for X_img in X_imgs_copy:
        X_img = tf.image.random_hue(X_img, 0.08)
        X_img = tf.image.random_saturation(X_img, 0.6, 1.6)
        X_img = tf.image.random_brightness(X_img, 0.05)
        X_img = tf.image.random_contrast(X_img, 0.7, 1.3)
        X_imgs_copy[i] = X_img
        labels.append(train_label_dictionary[img_ids[i]])
        i=i+1
    return X_imgs_copy, labels
    


def trained_and_evaluate_model(model):
    train_folder = "Train_Images"
    class_names = ['fruit_woodiness', 'fruit_brownspot', 'fruit_healthy']
    train_labels = list(csv.reader(open('Train.csv')))
    del train_labels[0]
    BATCH_SIZE = 40
    accuracies = []
    losses = []

    # create a dictionary of the training lables
    train_label_dictionary = {}
    for j in range(len(train_labels)):
        if train_labels[j][1] == class_names[0]:
            train_label_dictionary[train_labels[j][0]] = 0
        elif train_labels[j][1] == class_names[1]:
            train_label_dictionary[train_labels[j][0]] = 1
        elif train_labels[j][1] == class_names[2]:
            train_label_dictionary[train_labels[j][0]] = 2
        else:
            train_label_dictionary[train_labels[j][0]] = random.randrange(0, 3)


    del train_labels

    print('')
    print('========================================================================')
    print('Training')
    print('========================================================================')
    for i in range(50):
        start = i * BATCH_SIZE
        end = (i + 1) * BATCH_SIZE
        print('start: ', start)
        print('end: ', end)
        print("batch: ", i)

        # Get the train images with corresponding lables in batches
        train_images = []
        train_labels = []
        index = 0
        img_ids = []
        for filename in os.listdir(train_folder):
            if index in range(start, end):
                img = cv2.imread(os.path.join(train_folder, filename))
                if img is not None:
                    normalized_image = img / 255.0
                    train_images.append(normalized_image)
                    img_id = filename[0:filename.find('.')]
                    img_ids.append(img_id)
                    train_labels.append(train_label_dictionary[img_id])

            index = index + 1
        
        salt_and_pepper_images, salt_and_pepper_labels = add_salt_pepper_noise(train_images, img_ids, train_label_dictionary)
        gray_images, gray_labels = add_color_augmentation(train_images, img_ids, train_label_dictionary)

        for img in salt_and_pepper_images:
            train_images.append(img)

        for label in salt_and_pepper_labels:
            train_labels.append(label)

        for img in gray_images: 
            train_images.append(img)
        
        for label in gray_labels:
            train_labels.append(label)

        print('train images: ',len(train_images))
        print('train labels: ',len(train_labels))

        history = model.fit(np.array(train_images), np.array(train_labels), epochs=8, shuffle=True)
        accuracies.append(history.history['accuracy'])
        losses.append(list(history.history['loss']))

        del train_images
        del train_labels

    f = open("accuracies.csv", "a")
    i = 0
    accuracies = accuracies.flatten()
    for acc in accuracies:
        line = str(acc) + ',' + str(i) + '\n'
        f.write(line)
        i = i + 1
    f.close()

    f = open("losses.csv", "a")
    i = 0
    losses = losses.flatten()
    for loss in losses:
        line = str(loss) + ',' + str(i) + '\n'
        f.write(line)
        i = i + 1
    f.close()

    # # Evaluate
    print('')
    print('========================================================================')
    print('Evaluating')
    print('========================================================================')

    train_folder = "Train_Images"
    BATCH_SIZE = 100
    total_loss = 0
    total_acc = 0

    for i in range(20, 30):
        start = i * BATCH_SIZE
        end = (i + 1) * BATCH_SIZE
        print('start: ', start)
        print('end: ', end)
        print("batch: ", i)

        # Get the train images in batches
        test_images = []
        test_labels = []
        index = 0
        for filename in os.listdir(train_folder):
            if index in range(start, end):
                img = cv2.imread(os.path.join(train_folder, filename))
                if img is not None:
                    normalized_image = img / 255.0
                    test_images.append(normalized_image)
                    img_id = filename[0:filename.find('.')]
                    test_labels.append(train_label_dictionary[img_id])
            index = index + 1
        # print('test images: ',len(test_images))
        # print('test lables: ',len(test_labels))

        loss, acc = model.evaluate(np.array(test_images), np.array(test_labels), verbose=2)
        total_loss = total_loss + loss
        total_acc = total_acc + acc
        print('loss: ', loss)
        print('acc: ', acc)

        del test_images
        del test_labels

    avg_loss = total_loss / 10
    avg_acc = total_acc / 10
    print('avg loss: ', avg_loss)
    print('avg_acc', avg_acc)

if __name__ == "__main__":
    model = create_model()
    trained_and_evaluate_model = trained_and_evaluate_model(model)
