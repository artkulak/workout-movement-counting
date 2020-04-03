from Net import Net
import torch
import torchvision.transforms as transforms

import cv2
import numpy as np
import argparse
import os


class Utils:
    '''
    Class contains helper functions for the pipeline
    '''

    def __init__(self, isStream = False):
        self.isStream = isStream

    @staticmethod
    def getArgs():
        '''
        Reads arguments "type" and "file" from the command line
        :return: Returns the values of the read arguments
        '''

        parser = argparse.ArgumentParser()
        parser.add_argument('--file', help='Path to video file')
        args = parser.parse_args()

        return args.file

    @staticmethod
    def prepareModel(path):
        '''
        Loads the model and prepares it for inference
        :param path: path to the pytorch state dict file
        :return: model
        '''
        net = Net()
        net.load_state_dict(torch.load(path))
        net.eval()
        return net

    @staticmethod
    def prepareTransforms():
        '''
        Prepares transformers for image preprocessing
        :return: transformers for preprocessing
        '''
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
            ])

        return transform

    @staticmethod
    def contains(a, b):
        '''
        Checks if array b is a subsequence of array a
        :param a:
        :param b:
        :return: returns True if array b is a subsequence of a, else False
        '''
        for i in range(a.shape[0] - b.shape[0] + 1):
            if (a[i:i + b.shape[0]] == b).all():
                return True
        return False

    @staticmethod
    def readFrame(cap, IMG_SIZE=(128, 128)):
        '''
        Reads the next frame using cap from cv2.VideoCapture
        :param cap: cv2.VideoCapture instance
        :param IMG_SIZE: size of the returned image
        :return: Returns frame and its gray analog
        '''

        ret, origFrame = cap.read()
        frame = cv2.resize(origFrame, IMG_SIZE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return origFrame, frame, gray

    @staticmethod
    def getOptFlow(flow, prev_gray, gray, mask):
        '''
        Returns image with dense optical flow image
        :param flow: previous optical flow value
        :param prev_gray: previous frame gray image
        :param gray: current frame gray image
        :param mask:
        :return: opt flow image, new mask, new opt flow object
        '''
        if len(flow) == 0:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 100, 3, 7, 1.1, 0)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, flow, 0.5, 3, 100, 3, 7, 1.1, 0)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        return rgb, mask, flow

    def displaySteam(self, frame, moves=0):
        '''
        Displays a video stream of current actions and the total number of moves
        :param frame: current original frame
        :param moves: current total number of moves
        :return:
        '''
        output = cv2.resize(frame, (640, 480))
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (20, 20)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(output, str(moves),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        if not self.isStream:
            cv2.imshow("Stream", output)
        else:
            imgencode = cv2.imencode('.jpg', output)[1]
            stringData = imgencode.tostring()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')