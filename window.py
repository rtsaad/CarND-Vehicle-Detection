import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import time
from scipy.ndimage.measurements import label

# Import Project packages
import features

dist_pickle = pickle.load( open("classifier_pickle_rbf.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
hog_channel = dist_pickle["hog_channel"]
color_space = dist_pickle["color_space"]
spatial_feat =  dist_pickle["spatial_feat"]
hist_feat= dist_pickle["hist_feat"]
hog_feat= dist_pickle["hog_feat"]

## Class to keep track of cars
class HeatMap():

    n = 15
    def __init__(self, w, h):
        self.reset(w, h)

    def reset(self, w, h):
        self.box_n = []
        self.heat_w = w
        self.heat_h = h

    def add_heat(self, box_list):
        if len(self.box_n) > self.n:
            self.box_n.pop(0)
        self.box_n.append(box_list)

    def get_heatmap(self):
        heatmap = np.zeros((self.heat_h,self.heat_w)).astype(np.float)        
        # Iterate through list of bboxes
        for box_list in self.box_n:
            for box in box_list:                
                # Add += 1 for all pixels inside each bbox
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1                
            
        # Return thresholded map
        heatmap[heatmap <= len(self.box_n)*(2)] = 0     #1.1 

        return heatmap
        
heat_box = HeatMap(1280,720)
## Heatmap
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_bboxes(img, boxes):
    # Iterate through all detected cars
    for box in boxes:        
        # Draw the box on the image
        cv2.rectangle(img, box[0], box[1], (0,0,255), 6)
    # Return the image
    return img

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


## Window Search
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel='ALL', color_space='YCrCb', window=64, spatial_feat=True, hist_feat=True, hog_feat=True):

    if color_space!='YCrCb':
        if color_space=='HLS':
            conv='RGB2HLS'
        elif color_space=='HSV':
            conv='RGB2HSV'
        elif color_space=='LUV':
            conv='RGB2LUV'
        elif color_space=='YUV':
            conv='RGB2YUV'
        else:
            conv='RGB2'+color_space
    else:
        conv='RGB2YCrCb'
    
    draw_img = np.copy(img)
    #img = img.astype(np.float32)/255
    box_list = []
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = features.convert_color(img_tosearch, conv=conv)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = window
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    if hog_feat:
        if hog_channel=='ALL' or hog_channel==0:
            hog1 = features.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        if hog_channel=='ALL' or hog_channel==1:
            hog2 = features.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        if hog_channel=='ALL' or hog_channel==2:
            hog3 = features.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            if hog_feat:
                if hog_channel=='ALL' or hog_channel==0:
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                if hog_channel=='ALL' or hog_channel==1:
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                if hog_channel=='ALL' or hog_channel==2:
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

                if hog_channel=='ALL':
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                elif hog_channel==0:
                    hog_features = hog_feat1
                elif hog_channel==1:
                    hog_features = hog_feat2
                elif hog_channel==2:
                    hog_features = hog_feat3
            else:
                hog_features = []

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            if spatial_feat:
                spatial_features = features.bin_spatial(subimg, size=spatial_size)
            else:
                spatial_features = []

            if hist_feat:
                hist_features = features.color_hist(subimg, nbins=hist_bins)
            else:
                hist_features = []
            
            # Scale features and make a prediction            
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

            #Predict
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)                
                box_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return box_list

def check_image(img):

    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    
    ystart = 400
    ystop = 528
    scale = 1.2

    box_list1 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel, color_space=color_space, window=64, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    ystart = 500 
    ystop = 692
    scale = 1.5
    
    box_list2 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel, color_space=color_space, window=64, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    box_list = box_list1 + box_list2    
    
    # Apply threshold to help remove false positives
    heat_box.add_heat(box_list)
    heatmap = heat_box.get_heatmap()

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heatmap, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)    
    return draw_img

if __name__ == "__main__":
    img = cv2.imread('test_images/test6.jpg')    
    img = features.convert_color(img, conv='BGR2RGB')#cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#
    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    ystart = 400
    ystop = 528#528#656 #528
    scale = 1.2

    box_list1 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel, color_space=color_space, window=64, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    ystart = 500 
    ystop = 692
    scale = 1.5
    
    box_list2 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel, color_space=color_space, window=64, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    box_list = box_list1 + box_list2

    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    #draw_img = draw_bboxes(np.copy(img), box_list)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    features.plot_side_by_side(draw_img, heatmap, title1='Vehicle Detected', title2='HeatMap', filename='heatmap5')
    #features.plot_side_by_side(img, draw_img, title1='Input', title2='Windowns', filename='output')
    
