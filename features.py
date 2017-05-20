import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import random
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

def plot_side_by_side(img1, img2, title1=None, title2=None, filename=None):
    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(img1)#, cmap='gray')
    if title1 is not None:
        plt.title(title1)
    plt.subplot(122)
    plt.imshow(img2, cmap='gray')
    if title2 is not None:
        plt.title(title2)
    if filename is not None:
        plt.savefig('output_images/'+filename+'.png', dpi=100)
    plt.show()

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'BGR2RGB':        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)        
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = cv2.imread(file)        
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                            orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

def load_trainnig_data(num=None, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    #Load cars images
    cars = glob.glob('data/vehicles/**/*.png')
    #Shuffle data
    cars = random.sample(cars, len(cars))
    #get subset of values if asked
    if num is not None and num < len(cars):
        cars = cars[:num]
        
    #Load not cars images
    notcars = glob.glob('data/non-vehicles/**/*.png')
    #Shuffle data
    notcars = random.sample(notcars, len(notcars))
    #get subset of values if asked
    if num is not None and num < len(notcars):
        notcars = notcars[:num]
        
    #Get features
    car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size,
                                    hist_bins=hist_bins, hist_range=hist_range, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size,
                                    hist_bins=hist_bins, hist_range=hist_range, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    #Normalize features
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    return (len(cars), len(notcars), len(car_features), len(notcar_features),  X,scaled_X, X_scaler)

if __name__ == "__main__":        
    print("Load Images")
    color_space=  'YCrCb'
    spatial_size= (16,16)
    hist_bins=    64
    hist_range=   (0,256)
    orient=       8
    pix_per_cell= 8
    cell_per_block= 2
    hog_channel=  0#'ALL'
    spatial_feat= True
    hist_feat=    True
    hog_feat=     True
    cars, notcars, _, _, X, scaled_X,_  = load_trainnig_data(num=10, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)    
    print("Show Features")
    print("Cars: ", len(cars), " Non-Cars: ", len(notcars))

    car_ind = np.random.randint(0, len(cars))
    not_ind = np.random.randint(0, len(notcars))
    img = cv2.imread(cars[car_ind])
    img = convert_color(img, conv='BGR2RGB')

    img_ = cv2.imread(notcars[not_ind])
    img_ = convert_color(img_, conv='BGR2RGB')
    plot_side_by_side(img, img_, title1='Car', title2='Non-Car', filename='example')
    
    #img2 = convert_color(img, conv='BGR2YCrCb')
    
    #_, hog_img = get_hog_features(img2[:,:,0], orient, pix_per_cell, cell_per_block, 
    #                   vis=True, feature_vec=True)
    #plot_side_by_side(img, hog_img, title1='Original', title2='Hog', filename='hog_img')
    
    # Plot an example of raw and scaled features
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(cars[car_ind]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    plt.savefig('output_images/car_feature_all_feat.png', dpi=100)
    plt.show()    
    fig.tight_layout()

    not_ind = np.random.randint(0, len(notcars))
    print(X[len(cars) + car_ind].shape)
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(notcars[not_ind]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[len(cars) + not_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[len(cars) + not_ind])
    plt.title('Normalized Features')
    plt.savefig('output_images/notcar_feature_all_feat.png', dpi=100)
    plt.show()    
    fig.tight_layout()

    cor = np.correlate(scaled_X[car_ind], scaled_X[len(cars) + not_ind])
    print(cor)
    cor = np.correlate(scaled_X[car_ind], scaled_X[len(cars) + not_ind], 'same')
    
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Car')
    plt.subplot(132)
    plt.plot(scaled_X[len(cars) + not_ind])
    plt.title('Normalized Not Car')
    plt.subplot(133)
    plt.plot(cor)
    plt.title('Correlation')
    plt.savefig('output_images/correlate_feature_all_feat.png', dpi=100)
    plt.show()    
    fig.tight_layout()
