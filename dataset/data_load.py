from scipy import io
import numpy as np
import cv2

def load_hoda (train_size = 1000,test_size =200,resize =5 ):
    # load data
    dataset = io.loadmat("dataset/Data_hoda_full.mat")
    x_train = np.squeeze(dataset["Data"][:train_size])
    y_train = np.squeeze(dataset["labels"][:train_size])
    x_test = np.squeeze(dataset["Data"][train_size:train_size+test_size])
    y_test = np.squeeze(dataset["labels"][train_size:train_size+test_size])

    # resize
    x_train_resize = [cv2.resize(img,dsize=(resize,resize)) for img in x_train] 
    x_test_resize = [cv2.resize(img,dsize=(resize,resize)) for img in x_test] 

    # reshape
    X_train = np.reshape(x_train_resize, [-1,25])
    X_test = np.reshape(x_test_resize, [-1,25])

    return X_train,y_train,X_test,y_test