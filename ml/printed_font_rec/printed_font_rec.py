import os
import cv2
import csv
import time
import tarfile
import urllib.request
import numpy as np

#np.set_printoptions(threshold=np.inf)

LEARNING_RATE = 0.01
BLOCK_SIZE = 5
IMG_SIZE = 5
DATASET_LINK = 'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz'
LABEL_ELEMENTS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
                  'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                  'y', 'z']
repo_path = os.path.dirname(os.path.abspath(__file__))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def get_dataset(url):
    print("downloading dataset file...")
    filename = os.path.basename(url)
    urllib.request.urlretrieve(url, os.path.join(repo_path, filename))
    print("dataset file is {} at {}".format(filename, repo_path))
    
    print("extracting files...")
    tar = tarfile.open(os.path.join(repo_path, filename))
    tar.extractall(path = repo_path)
    tar.close()

def create_csv(csv_file_name):
    print("creaing csv file...")
    with open(os.path.join(repo_path, csv_file_name), 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        index = 0
        for root, dirs, files in os.walk(os.path.join(repo_path, 'English/Fnt')):
            for dirname in dirs:
                dir_path = os.path.join(root, dirname)
                for filename in os.listdir(dir_path):
                    csv_row = [os.path.join(dir_path, filename), LABEL_ELEMENTS[index]]
                    writer.writerow(csv_row)
                index = index + 1

def train_1(csv_file_name, trained_class):
    print("start training")
    with open(os.path.join(repo_path, csv_file_name), 'r', newline = '') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        '''
        initiate X as the input of the neural network
        X's size = N * (IMG_SIZE * IMG_SIZE) where N = number of images
        '''        
        #N = sum(1 for row in reader if row[1] == trained_class)
        N = BLOCK_SIZE
        X = np.zeros((N, IMG_SIZE * IMG_SIZE))
        # initiate y as the standard output of the neural network
        y = np.zeros((N, 1))
        # initiate W1 as weights between l1 and l2
        W1 = 2 * np.random.random((IMG_SIZE * IMG_SIZE, IMG_SIZE * IMG_SIZE + 1)) - 1
        # initiate W2 as weights between l2 and l3
        W2 = 2 * np.random.random((IMG_SIZE * IMG_SIZE + 1, 1)) - 1
        
        counter = 0
        for row in reader:
            if row[1] == trained_class and counter < BLOCK_SIZE:
                img = cv2.imread(row[0], cv2.IMREAD_GRAYSCALE)
                res = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_CUBIC)
                img_arr = np.asarray(res)
                '''
                setting pixel array to binary(optional)
                the img array remains in 256 form after binary transformation with cv2.threshold
                ret, bi_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                '''
                img_arr[img_arr < 128] = 0
                img_arr[img_arr > 127] = 1
                X[counter, :] = img_arr.reshape(1, IMG_SIZE * IMG_SIZE)
                y[counter] = LABEL_ELEMENTS.index(row[1])
                counter += 1
                y[:, 0] = 0.8
        
        for num_iter in range(100000000):
            err_num = 0
            # set layer1 as input layer
            # hidden layer
            l2 = sigmoid(np.dot(X, W1))
            # ouput layer
            l3 = sigmoid(np.dot(l2, W2))
            # layer3 derivative
            l3_d = (y - l3) * sigmoid_derivative(l3)
            # layer2 derivative
            l2_d = np.dot(l3_d, W2.T) * sigmoid_derivative(l2)
            # update weights
            W2 += LEARNING_RATE * np.dot(l2.T, l3_d)
            W1 += LEARNING_RATE * np.dot(X.T, l2_d)
            
            for index in range(N):
                if abs(l3[index] - y[index]) > 0.001:
                   err_num += 1

            print("iter:", num_iter, "err:", err_num, l3.T[0: N], end = '\r')
            if err_num == 0:
                print("\ntraining finished")
                str = input("input 'yes' to print weights, or anything else to exit\n: ")
                if(str == 'yes'):
                    print("W1:\n", W1)
                    print("W2:\n", W2)
                #print("saving weights array")
                break
            #time.sleep(0.1)


if __name__ == "__main__":
    csv_file_name = 'printed_font_rec.csv'
    trained_class = 'A'
    get_dataset(DATASET_LINK)
    create_csv(csv_file_name)
    train_1(csv_file_name, trained_class)
