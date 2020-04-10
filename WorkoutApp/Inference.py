import cv2
import numpy as np
from Utils import Utils
import os

import threading
from time import time



class ExerciseCapture:
    # Class for capturing the concrete exercise within time/move limit from stream or video input

    def __init__(self, model_path = 0, fromStream = True, timeWise = True, thresh = 10, name = 'None'):
        self.takeFrame = 1
        self.IM_SIZE = (128, 128)

        self.model_path = model_path
        self.fromStream = fromStream
        self.timeWise = timeWise
        self.thresh = thresh

        self.utils = Utils(self.fromStream)
        self.name = name


        # loading model and image transforms
        if self.model_path != 0:
            self.net = self.utils.prepareModel(self.model_path)
        self.transform = self.utils.prepareTransforms()

        self.origFrame, self.moves, self.totalTime = cv2.imread('blank.png'), 0, 0

        if self.fromStream:
            self.file = 0
        else:
            self.file = self.utils.getArgs()


    def classifyFrame(self, flow):
        '''
        Classify optical flow image with CNN
        :param net: CNN model
        :param transform: transforms for image preprocessing
        :param flow: optical flow image
        :return: Class of the input optical flow image
        '''
        cv2.imwrite(f'''flow.png''', flow)
        rgb = cv2.imread('flow.png')
        image = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        x = self.transform(image).reshape(1, 3, self.IM_SIZE[0], self.IM_SIZE[1])
        try:
            os.remove('flow.png')
        except:
            pass

        return np.argmax(self.net(x).detach().numpy())

    def getMovesCount(self, labels, moves, sequence = [0, 0, 0, 2, 2, 2]):
        '''
        If the current label sequence contains the move label sequence clear label sequence and add 1 push up to the moves
        counter
        :param labels: Current CNN prediction sequence
        :param moves: Current number of moves
        :param sequence: The move sequence to search for in the labels sequence
        :return: current label sequnce and total move count
        '''
        if self.utils.contains(np.array(labels)[np.array(labels) != 1], np.array(sequence)):
            if self.name == 'Sit up':
                moves += 1
            else:
                moves += 1
            labels = [1, 1, 1, 1, 1]
        return labels, moves

    def hasFinished(self, timePassed, moves):
        '''
        Displays if the current move is finished or not
        :param timePassed: time since move start
        :param moves: total number of moves since start
        :return: isMoveFinished?
        '''
        if self.timeWise:
            if timePassed >= self.thresh:
                return True
        else:
            if moves >= self.thresh:
                return True

        return False
  
    def runPipeline(self, cap):
        '''
        Runs the counting pipeline for the concrete exercise
        :return:
        '''

        t = threading.currentThread()
        self.origFrame, first_frame, prev_gray = self.utils.readFrame(cap, self.IM_SIZE)

        mask = np.zeros_like(first_frame)
        mask[..., 1] = 255

        self.moves, frameIndex = 0, 0
        flow = []
        labels = [1, 1, 1, 1, 1]

        startTime = time()

        while cap.isOpened():
            self.origFrame, _, gray = self.utils.readFrame(cap, self.IM_SIZE)
            if not t.do_run:
                self.origFrame = cv2.imread('blank.png')
                break

            # getting optical flow image for the current frame
            rgb, mask, flow = self.utils.getOptFlow(flow, prev_gray, gray, mask)

            if frameIndex % self.takeFrame == 0 and frameIndex > 0:
                # classifying current frames' optical flow image
                labels.append(self.classifyFrame(rgb))

                # updates the labels sequence and moves count
                labels, self.moves = self.getMovesCount(labels, self.moves)

            prev_gray = gray

            # shows the current video stream
            #self.utils.displaySteam(self.origFrame, self.moves)

            frameIndex += 1

            self.totalTime = time() - startTime

            if self.hasFinished(self.totalTime, self.moves):
                return self.moves, self.totalTime

            if cv2.waitKey(20) & 0xFF == ord('q'):
                return self.moves, self.totalTime
        


if __name__ == '__main__':
    ex = ExerciseCapture('model.pt', True, False, 10)
    moves, totalTime = ex.runPipeline()
    print(moves, totalTime)

