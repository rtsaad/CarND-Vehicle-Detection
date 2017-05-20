import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#Project imports
import features
import window

def process_image(img):
    draw_img = window.check_image(img)
    return draw_img 

#Flags
import argparse
parser = argparse.ArgumentParser(description='Vehicle Detection.')
parser.add_argument('video_input', type=str, metavar='project_video', help="Video to process")
parser.add_argument('video_output', type=str, metavar='track', help="file output")

#Resume pipeline
def pipeline():
    args = parser.parse_args()
    white_output = 'output_images/' + args.video_output + '.mp4'
    clip1 = VideoFileClip(args.video_input + ".mp4")
    white_clip = clip1.fl_image(process_image) 
    white_clip.write_videofile(white_output, audio=False)

##Execute
pipeline()
