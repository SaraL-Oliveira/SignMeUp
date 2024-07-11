import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, log_loss
from scipy.interpolate import interp1d
import pickle
from sklearn import tree


#import pandas as pd

from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier

#TODO: Find a way to store the max_sample_size of the trainning data (maybe store in a params file)
#This comes from the training of the model and is needed because test data and trainning have to be of the same size
MAX_SAMPLE_SIZE = 1968 #1836 # 1044 #1164 #1152 #912 #1152 #546  
#WORDS = ['coffee', 'hello', 'want']


def read_trainning_data(filenames, num_files):
    trainning_data = []
    max_size_of_sample = 0
    for filename in filenames:
        for i in range(num_files):
            data_matrix = []
            path = filename + str(i) + ".txt"
            f = open(path, "r")
            for line in f:
                line = line.replace('\n', '')
                data = line.split(',')
                data = list(map(int, data[1:]))
                for val in data:
                    data_matrix.append(val)
            trainning_data.append(data_matrix)
            if(len(data_matrix) > max_size_of_sample):
                max_size_of_sample = len(data_matrix)
    return trainning_data, max_size_of_sample

def prepare_trainning_data(trainning_data, num_files_per_set):
    train_y = [0,0,0,0,0,0,1,1,1,1,1,1]
    train_x = []

    for set_idx in range(len(trainning_data)):
        for data_matrix_idx in range(num_files_per_set):
            train_x.append(trainning_data[set_idx][data_matrix_idx])

    return train_x, train_y

def read_test_data_for_report(filenames, num_files, start_idx):
    test_data = []
    for filename in filenames:
        for i in range(start_idx, start_idx + num_files):
            data_matrix = []
            path = filename + str(i) + ".txt"
            f = open(path, "r")
            for line in f:
                line = line.replace('\n', '')
                data = line.split(',')
                data = list(map(int, data[1:]))
                for val in data:
                    data_matrix.append(val)
            test_data.append(data_matrix)
    return test_data

# Adds rows to each gesture image so they all have 40 rows (matrix 40x6)
def add_rows_to_max(trainning_data, max_size):
    trainning_data_upd = []
    for gesture in trainning_data:
        if len(gesture) < max_size:
            #for i in range(max_size - len(gesture)):
            #    gesture.append(0)
            gesture = interpolate(gesture, max_size)
        trainning_data_upd.append(gesture)
    return trainning_data_upd

# Here, instead of adding I should do the average size accross all samples and either remove or add for each case (I wrote this in the report)
def add_rows_to_max_test(test_x, max_size):
    test_x_upd = test_x
    if len(test_x) < max_size:
        #for i in range(max_size - len(test_x)):
        #    test_x_upd.append(0)
        test_x_upd = interpolate(test_x, max_size)
    elif len(test_x) > max_size:
        half = (len(test_x) - max_size)//2
        test_x_upd = test_x[half:-half]
        
    return test_x_upd

def print_list(list_to_print):
    for i in range(len(list_to_print)):
        for x in range(len(list_to_print[i])):
            print(list_to_print[i][x])
        print("\n\n")

def train_model(words, path_to_folder, num_train_samples):
    #prepare a list of all filepaths to be used to trainning
    filenames = []
    train_y = []
    for word in words:
        filenames.append("words_train_50/" + word + "_split/" + word + "_split")
        train_y.extend([word] * num_train_samples)

    train_x, max_size = read_trainning_data(filenames, num_train_samples)
    train_x = add_rows_to_max(train_x, max_size)

    #clf = tree.DecisionTreeClassifier(criterion='gini')
    clf = svm.SVC(probability=True)
    #clf = MLPClassifier(random_state=1, max_iter=200)
    clf = clf.fit(train_x, train_y)

    return clf, max_size


def read_trainning_data_2hands(filenames, num_files):
    trainning_data = []
    max_size_of_sample = 0
    for filename in filenames:
        for i in range(num_files):
            data_matrix = []
            path_left = filename + "left_" + str(i) + ".txt"
            path_right = filename + "right_" + str(i) + ".txt"
            f_left = open(path_left, "r")
            f_right = open(path_right, "r")
            for line_left, line_right in zip(f_left, f_right):
                line_left = line_left.replace('\n', '')
                data_left = line_left.split(',')
                data_left = list(map(int, data_left)) #antes estava assim data_left[1:]
                line_right = line_right.replace('\n', '')
                data_right = line_right.split(',')
                data_right = list(map(int, data_right))
                for i in range(len(data_left)):
                    data_matrix.append(data_left[i])
                    data_matrix.append(data_right[i])
            trainning_data.append(data_matrix)
            if (len(data_matrix) > max_size_of_sample):
                max_size_of_sample = len(data_matrix)
    return trainning_data, max_size_of_sample

def normalize_each_sign_test(test_x):
    sign = []
    min_val = min(test_x)
    max_val = max(test_x)
    for val in test_x:
        sign.append((val - min_val)/(max_val - min_val))
    return sign

def read_test_data_2hands(filepath, sample):
    test_data = []
    data_matrix = []
    path_left = filepath + "left_" + str(sample) + ".txt"
    path_right = filepath + "right_" + str(sample) + ".txt"
    f_left = open(path_left, "r")
    f_right = open(path_right, "r")
    for line_left, line_right in zip(f_left, f_right):
        line_left = line_left.replace('\n', '')
        data_left = line_left.split(',')
        data_left = list(map(int, data_left))
        line_right = line_right.replace('\n', '')
        data_right = line_right.split(',')
        data_right = list(map(int, data_right))
        for i in range(len(data_left)):
            data_matrix.append(data_left[i])
            data_matrix.append(data_right[i])
    return data_matrix

def normalize_each_sign(train_x):
    new_train_x = []
    for sign in train_x:
        min_val = min(sign)
        max_val = max(sign)
        new_sign = []
        for val in sign:
            new_sign.append((val - min_val)/(max_val - min_val))
        new_train_x.append(new_sign)
    return new_train_x


def train_model_2hands(words, path_to_folder, num_train_samples):
    #prepare a list of all filepaths to be used to trainning
    filenames = []
    train_y = []
    for word in words:
        filenames.append(path_to_folder + word + "/Split/")
        train_y.extend([word] * num_train_samples)

    train_x, max_size = read_trainning_data_2hands(filenames, num_train_samples)
    train_x = add_rows_to_max(train_x, max_size)
    train_x = normalize_each_sign(train_x)

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    #clf = svm.SVC(probability=True)
    #clf = MLPClassifier(random_state=1, max_iter=50)
    clf = clf.fit(train_x, train_y)

    return clf, max_size

# Used for testing each model return accuracy metrics
def test_model(words, path_to_folder, num_train_samples, start_idx, model, max_size):
    #prepare a list of all filepaths to be used to trainning
    filenames = []
    test_y = []
    for word in words:
        filenames.append("words_train_50/" + word + "_split/" + word + "_split")
        test_y.extend([word] * num_train_samples)
    test_x = read_test_data_for_report(filenames, num_train_samples, start_idx)
    predictions = []
    predictions_proba = []
    for test in test_x:
        test = np.array(add_rows_to_max_test(test, max_size))
        test = test.reshape(1, -1)
        predict = model.predict(test)
        predict_proba = model.predict_proba(test)
        predictions.append(str(predict[0]))
        predictions_proba.append(predict_proba[0])
    
    print("Accuracy Score: " + str(accuracy_score(test_y, predictions)))
    print("Confusion Matrix: " + str(confusion_matrix(test_y, predictions)))
    print("Precision: " + str(precision_score(test_y, predictions, average=None)))
    print("Recall: " + str(recall_score(test_y, predictions, average=None)))
    print("F1 Score: " + str(f1_score(test_y, predictions, average=None)))
    print("Log Loss: " + str(log_loss(test_y, predictions_proba)))
    return 0
        



def predict_with_filename(filepath, model, words, max_size, sample):
    test_x = read_test_data_2hands(filepath, sample)
    test_x = add_rows_to_max_test(test_x, max_size)
    test_x = normalize_each_sign_test(test_x)
    test_x = np.array(test_x)
    test_x = test_x.reshape(1, -1)
    
    predict = model.predict(test_x)
    print(predict)
    return predict


# Functions for 2 hands

# https://stackoverflow.com/questions/29085268/resample-a-numpy-array
# Receives a numpy array and interpolates
def interpolate(arr, target_size):
    resample = interp1d(np.linspace(0,1, len(arr)), arr, 'linear')
    return resample(np.linspace(0,1, target_size))

# Creates a list with 12 vals for one instant, then 12 next, and 12 next and so on
def read_test_data(sign_left, sign_right):
    test_data = []
    for left, right in zip(sign_left, sign_right):
        print(left)
        print("\n")
        left = (left.replace('\n', '')).split(',')
        right = (right.replace('\n', '')).split(',')
        left = list(map(int, left[1:]))
        right = list(map(int, right[1:]))
        for val in left:
            test_data.append(val)
        for val in right:
            test_data.append(val)
    return test_data


def read_test_data_real_time(sign_left, sign_right):
    test_data = []
    for left, right in zip(sign_left, sign_right):
        for val in left:
            test_data.append(val)
        for val in right:
            test_data.append(val)
    return test_data


def predict(model, sign_left, sign_right, max_size):
    #test_x = read_test_data(sign_left, sign_right)
    test_x = read_test_data_real_time(sign_left, sign_right)
    test_x = add_rows_to_max_test(test_x, max_size)
    test_x = normalize_each_sign_test(test_x)
    test_x = np.array(test_x)
    test_x = test_x.reshape(1, -1)
    
    predict = model.predict(test_x)
    return predict


def load_model(model_filepath):
    model = pickle.load(open(model_filepath, "rb"))
    return model

if __name__ == '__main__':
    words = ['ACABAR', 'CAFE', 'CHOCOLATE', 'COMER', 'DEPOIS', 'EU', 'GOSTO', 'MENU', 'MUITO', 'POR_FAVOR', 'POSSO', 'TOMAR']
    #words = ['ACABAR', 'CAFE', 'COMER', 'EU', 'GOSTO', 'MUITO', 'POR_FAVOR', 'POSSO', 'TOMAR']
    words = ['CHOCOLATE', 'EU', 'GOSTO']
    path_to_folder = "TrainningSet_CleanV2/"
    NUM_TRAIN_SAMPLES = 25
    MAX_SAMPLE_SIZE = 1968 #1836 #1056 #1152 #546

    
    model, max_size_sample = train_model_2hands(words, path_to_folder, NUM_TRAIN_SAMPLES)
    print(max_size_sample)
    pickle.dump(model, open("decision_tree_clean_3words", "wb"))
    

    #score = test_model(words, path_to_folder, 8, 40, model, max_size_sample)

    #TODO: I should normalize the raw data here! Use the same function as in active_segmentation.py
    
    #TO PREDICT ONE SPECIFIC WORD
    """
    test_filepath = "TrainningSet_3Words/EU/Split/"
    clf = pickle.load(open("decision_tree_clean_3words", "rb"))
    predict = predict_with_filename(test_filepath, clf, words, MAX_SAMPLE_SIZE, 25)
    print(predict)
    """

    

    #model, history = setup_model()
    #model.save('my_classifier.keras')