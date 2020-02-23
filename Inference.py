import cv2
import numpy as np
import pandas as pd
import argparse
import pickle

MODEL_PATH = 'model.pkl'

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=1000, qualityLevel = 0.0011, minDistance = 1, blockSize = 3)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
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
    preds = []
    model = pickle.load(open(MODEL_PATH, 'rb'))
    fileType, file = getArgs()
    cap = cv2.VideoCapture(file)
    color = (0, 255, 0)

    ret, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    prev = cv2.goodFeaturesToTrack(image = prev_gray, mask = None, **feature_params)
    good_old = [0]*feature_params['maxCorners']
    k = 0
    pushes = 0
    nMax = 50
    data = pd.DataFrame(columns = [f'xval{i}' for i in range(nMax)] + [f'yval{i}' for i in range(nMax)])
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(good_old) < 0.4 * feature_params['maxCorners']:
            print('Dropping corners due to insufficient number...')
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev = cv2.goodFeaturesToTrack(image=prev_gray, mask=None, **feature_params)

        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)

        # Selects good feature points for previous position
        good_old = prev[status == 1]
        # Selects good feature points for next position
        try:
            good_new = next[status == 1]
        except:
            continue

        # Draws the optical flow tracks
        dxs, dys = [], []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            if k%5==0 and k>0:
                dxs.append(a-c)
                dys.append(b-d)
            frame = cv2.circle(frame, (a, b), 3, color, -1)

        if k%5 == 0 and k > 0:
            if len(dxs) < nMax:
                while len(dxs) < nMax:
                    dxs.append(0)

            if len(dys) < nMax:
                while len(dys) < nMax:
                    dys.append(0)

            dys = sorted(dys, key=lambda x: np.abs(x))
            dxs = sorted(dxs, key=lambda x: np.abs(x))

            data.loc[data.shape[0], nMax:2*nMax] = dxs[-nMax:]
            data.loc[data.shape[0]-1, :nMax] = dys[-nMax:]
            pred = model.predict(data.iloc[-1:, :])
            preds.append(pred[0])
            try:
                if preds[-1] == 1 and preds[-2] == 0:
                    pushes += 1
            except:
                pass


        # Overlays the optical flow tracks on the original frame
        output = frame

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (20, 20)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(output, str(pushes),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        # Updates previous frame
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)


        cv2.imshow("sparse optical flow", output)
        k += 1
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

