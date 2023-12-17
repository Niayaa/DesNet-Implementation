import sys
import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import logging
import time
import psutil
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from utils import get_dataset, divided_dataset
from sklearn.model_selection import train_test_split
from densenetModel import Model,CustomCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping



#load data from the utils
data_list = get_dataset()

# Function to get current memory usage
def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB
    #initialize
if __name__ == '__main__':
    train = True
    test = True
    image_size = (195, 195)
    data_type = "poly"
    sample_rate = 0.25
    image_size=(195,195)
    learning_rate=0.001
    epoch_num=30
    #get training and testing dataset
    data_train,data_test,label_train,label_test=divided_dataset(data_list,data_type,sample_rate,image_size)
if train:
    data_train = np.array(data_train)
    label_train = np.array(label_train)
    #make labels as one hot code for classification
    label_train = to_categorical(label_train, num_classes=4)
    #add one more channel since the image is grayscale
    data_train = np.expand_dims(data_train, axis=-1)
    #record precessing time and usage
    start_time = time.time()
    initial_memory = get_memory_usage()

    #Image Augmentation
    datagen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.5,
    height_shift_range=0.5,
    shear_range=0.5,
    zoom_range=0.5,
    )
    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()
    #Generate the augmentations
    train_generator = train_datagen.flow(data_train, label_train, batch_size=32)
    val_generator = val_datagen.flow(data_train, label_train, batch_size=32)
    #Set the value for optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    #build new model
    model = Model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #lord the record of training process into the logs file
    log_file = 'logs.txt'
    with open(log_file, 'a') as file:
        file.write("\n===== TRAIN Part =====\n")
    custom_callback = CustomCallback(log_file=log_file)
    #early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    model.fit(train_generator, epochs=epoch_num, validation_data=val_generator, verbose=1, callbacks=[custom_callback])
    #training process ended
    end_time = time.time()
    final_memory = get_memory_usage()
    training_time = end_time - start_time
    memory_used = final_memory - initial_memory
    with open(log_file, 'a') as file:
        file.write(f'Training - Time: {training_time:.2f} seconds - Memory used: {memory_used:.2f} MB\n')
#Test process
if test:
    data_test = np.array(data_test)
    label_test = np.array(label_test)
    #same operation as training process
    one_hot_label_test = to_categorical(label_test, num_classes=4)
    data_test = np.expand_dims(data_test, axis=-1)
    #record test processing time
    start_time = time.time()
    initial_memory = get_memory_usage()
    #get loss fuction and accuracy from evaluating the model
    loss, accuracy = model.evaluate(data_test, one_hot_label_test)
    #evaluating progress ended
    end_time = time.time()
    final_memory = get_memory_usage()
    #print(f"Test loss: {loss}")
    #print(f"Test accuracy: {accuracy}")

    evaluation_time = end_time - start_time
    memory_used = final_memory - initial_memory
    #lord the testing progress into the logs file
    log_file = 'logs.txt'
    with open(log_file, 'a') as file:
        file.write("\n===== TESTNG Part =====\n")
        file.write(f"Data Type: {data_type}\n")
        file.write("Depth: 8, growth_rate=24")
        file.write(f'Evaluation - Time: {evaluation_time:.2f} seconds - Memory used: {memory_used:.2f} MB\n')
    #get the prediction
    predictions = model.predict(data_test)
    #print(predictions)
    #processing the data of prediction and original to get ready for compare
    predicted_classes = np.argmax(predictions, axis=1)
    label_test = to_categorical(label_test, num_classes=4)
    true_classes = np.argmax(label_test, axis=1)
    #get report by comparing the prediction with true value
    report=classification_report(true_classes, predicted_classes)
    confusion_matrix=confusion_matrix(true_classes, predicted_classes)
    #print(report)
    #print(confusion_matrix)
    #lord the comparsing result into the logs data
    with open(log_file, 'a') as file:
        file.write(f"Test Loss: {loss}, Test Accuracy: {accuracy}\n")
        file.write("Classification Report:\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(str(confusion_matrix))
#produce the image result
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9    #pick 9 images as result

    #put image as 3 row * 3 column
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3) #set space

    for i, ax in enumerate(axes.flat):

        ax.imshow(images[i].reshape(195,195), cmap='binary')


        if cls_pred is None:
            xlabel = f"True: {cls_true[i]}"
        else:
            xlabel = f"True: {cls_true[i]}, Pred: {cls_pred[i]}" #set label

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

#load data of image
images = data_test[:9]
cls_true = true_classes[:9]
cls_pred = predicted_classes[:9]

#present image
plot_images(images, cls_true, cls_pred)
