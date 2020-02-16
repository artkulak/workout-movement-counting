import cv2 as cv
import numpy as np
import time
import argparse

import shutil
import os
from pyoptflow import LucasKanade

def getArgs():
    '''
    Reads arguments "type" and "file" from the command line
    :return: Returns the values of the read arguments
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='Is it train/valid/test video?')
    parser.add_argument('--file', help='Path to video file')
    args = parser.parse_args()

    return args.type, args.file

if __name__ == '__main__':

    fileType, file = getArgs()

    folders = {'flow': f'OptFlow{fileType}/',
               'frames': f'Frames{fileType}/'}

    shutil.rmtree(folders['flow'], ignore_errors=True)
    shutil.rmtree(folders['frames'], ignore_errors=True)
    os.mkdir(folders['flow'])
    os.mkdir(folders['frames'])

    cap = cv.VideoCapture(file)
    IMG_SIZE = (128, 128)

    ret, first_frame = cap.read()
    first_frame = cv.resize(first_frame, IMG_SIZE)

    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv.resize(frame, IMG_SIZE)
        cv.imshow("input", frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 50, 3, 5, 1.1, 0)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

       # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        cv.imshow("dense optical flow", rgb)
        if i % 10 == 0:
            cv.imwrite(f'''{folders['flow']}/{i}.png''', rgb)
            cv.imwrite(f'''{folders['frames']}/{i}.png''', frame)
        i+=1
        prev_gray = gray
        # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()