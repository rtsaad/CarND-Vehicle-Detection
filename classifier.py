import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

#Import Project packages
import features

#Define Features Parameters
color_space=  'YCrCb'
spatial_size= (16,16)
hist_bins=   64
hist_range=   (0,256)
orient=       8
pix_per_cell= 8
cell_per_block= 2
hog_channel=  0
spatial_feat= True
hist_feat=    True
hog_feat=     True
num =         None

#Perform a grid Search or train a SVM classifier
grid_search = True
#Select kernel: linear, rbf or decision
kernel = 'decision'
#Set the parameters below if grid search is False
C = 100 

max_depth = 10         # For Decision Tree only
min_samples_split = 10 # For Decision Tree only

## Load Training Data
print("Load Training Data")
t=time.time()
cars, notcars, car_features, notcar_features, X, scaled_X, X_scaler = features.load_trainnig_data(num=num, color_space=color_space, spatial_size=spatial_size,
                                    hist_bins=hist_bins, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to load data features...')
print('Load ', cars+notcars, ' images features, divided into ', cars, '/', notcars)
#Define the labels vector
y = np.hstack((np.ones(car_features), np.zeros(notcar_features)))

## Prepare Training Data
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
print('Training Data Splited: 80% / 20%')
print('Feature vector length:', len(X_train[0]))

## Training Classifier
if grid_search:
    #Use gridsearch to opmise parameters
    if kernel!='decision':
        parameters = {'C':[0.1, 1, 10, 100]}
        svc = svm.SVC(kernel= kernel)
    else:
        parameters={'min_samples_split' : [10,100,500],'max_depth': [1,10,20]}
        svc=DecisionTreeClassifier()
    clf = GridSearchCV(svc, parameters, n_jobs=4, verbose=2)
    # Check the training time for the SVC
    print('Start Grid Search for Classifier')
    t=time.time()
    clf.fit(X_train, y_train)
    print('Found best coeficients', clf.best_params_)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to find best coeficients...')
    #Load best coeficients
    if kernel!='decision':
        C = clf.best_params_['C']
    else:
        max_depth = clf.best_params_['max_depth']
        min_samples_split = clf.best_params_['min_samples_split']

#Train with best coeficients
print("Start Training Classifier")
if kernel!='decision':
    svc = svm.SVC(kernel=kernel, C=C)
else:
    svc = DecisionTreeClassifier()#max_depth=max_depth, min_samples_split=min_samples_split)
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
print('Save parameters')
output = open('classifier_pickle.p', 'wb')
data = {'svc':             svc,
        'scaler':          X_scaler,
        'orient':          orient,
        'pix_per_cell':    pix_per_cell,
        'cell_per_block':  cell_per_block,
        'spatial_size':    spatial_size,
        'hist_bins':       hist_bins,
        'color_space':     color_space,
        'hist_range':      hist_range,
        'hog_channel':     hog_channel,
        'spatial_feat':    spatial_feat,
        'hist_feat':       hist_feat,
        'hog_feat':        hog_feat,
        'num':             num
}
pickle.dump(data, output)
print('Training Finished')
