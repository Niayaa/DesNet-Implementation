import os
import numpy as np
import random
import tensorflow as tf
import cv2
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.utils import to_categorical
from PIL import Image

#This function is produced by the ELPV dataset provider
#Ref:
#ELPV Dataset. A Benchmark for Visual Identification of Defective Solar Cells in Electroluminescence Imagery.
#https://github.com/zae-bayern/elpv-dataset
def get_dataset(fname=None):
    if fname is None:
        # Assume we are in the utils folder and get the absolute path to the
        # parent directory.
        fname = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir))
        fname = os.path.join(fname, 'DenseNet/labels.csv')

    data = np.genfromtxt(fname, dtype=['|S19', '<f8', '|S4'], names=[
                         'path', 'probability', 'type'])
    image_fnames = np.char.decode(data['path'])
    probs = data['probability']
    types = np.char.decode(data['type'])

    def load_cell_image(fname):
        with Image.open(fname) as image:
            return np.asarray(image)

    dir = os.path.dirname(fname)

    images = np.array([load_cell_image(os.path.join(dir, fn))
                       for fn in image_fnames])

    data_list = [] #put into a datalist
    for i in range(len(images)):

        data_list.append([images[i], probs[i], types[i]])
    return data_list

#Preprocessing the data
def divided_dataset(data_list,data_type,sample_rate,image_size):
    data_set = []
    image_set = []
    label_set = []
    labels=[0,1, 0.3333333333333333, 0.6666666666666666]
    label_to_int = {label: index for index, label in enumerate(labels)} # make labels as [0,1,2,3]
    for data in data_list:
        #print(data[2])
        if data_type=="Both": #load both of poly and mono
            data_set.append(data)
            image = cv2.resize(data[0], image_size, interpolation=cv2.INTER_AREA)
            image = (image - np.min(image)) / (np.max(image) - np.min(image))#Normalization
            image_set.append(image)
            label_set.append(data[1])
        elif data[2]!=data_type:
            continue
        else:
            data_set.append(data)
            image = cv2.resize(data[0], image_size, interpolation=cv2.INTER_AREA)
            image = (image - np.min(image)) / (np.max(image) - np.min(image))#Normalization
            image_set.append(image)
            label_set.append(data[1])
            #print(data[2])
    #one_hot_labels = to_categorical(label_set, num_classes=4)
    image_set = np.array(image_set)
    label_set = [label_to_int[label] for label in label_set] # Making labels as int coding
    label_set = np.array(label_set)
    if len(image_set)>0 and len(label_set)>0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_rate, random_state=None)
        """
        :n_splits: the number of spliting the dataset
        test_size= 0.25: the test set accounts for 25% of the entire data set
        random_state: repeatability of results
        """
        for train_index, test_index in sss.split(image_set, label_set):
            X_train, X_test = image_set[train_index], image_set[test_index]
            y_train, y_test = label_set[train_index], label_set[test_index]

    #check the distrubtion of each data
    train_perfect=0
    train_damaged=0
    train_possibly=0
    train_likely=0
    test_perfect=0
    test_damaged=0
    test_possibly=0
    test_likely=0
    for i in y_train:
        if i == 0:  #full functional cell
            train_perfect+=1
        elif i==1:  #perfect
            train_damaged+=1
        elif i==2:  #probably
            train_possibly+=1
        else:       #likely
            train_likely+=1
    for i in y_test:    #same as training dataset
        if i==0:
            test_perfect+=1
        elif i==1:
            test_damaged+=1
        elif i==2:
            test_possibly+=1
        else:
            test_likely+=1
    #print(X_train)
    #print(y_train)
    #print("The train images:",len(X_train))
    #print("The test images:",len(X_test))
    #print("The train dataset labels:",len(y_train))
    #print("The test dataset labels:",len(y_test))
    #print the disrtibution of dataset
    print("\tTrainSet: %d \tTestSet: %d"%(len(y_train),len(y_test)))
    print("Perfect: \t%d \t\t%d"%(train_perfect,test_perfect))
    print("Damaged: \t%d \t\t%d"%(train_damaged,test_damaged))
    print("Possibly: \t%d \t\t%d"%(train_possibly,test_possibly))
    print("Likely: \t%d \t\t%d"%(train_likely,test_likely))
    return X_train, X_test, y_train, y_test
