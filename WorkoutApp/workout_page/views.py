from django.shortcuts import render, render_to_response, redirect
from django.http import HttpResponse,StreamingHttpResponse, HttpResponseServerError, JsonResponse
from django.views.decorators import gzip
import cv2
import time

import json

import numpy as np

import threading

from .models import Workouts

from Inference import ExerciseCapture
from workout import Workout
from time import time
from time import sleep


from warnings import filterwarnings
filterwarnings('ignore')


def get_frame():

    '''
    Yields the current frame from the workout.ex class
    '''


    while True:
        try:
            frame, moves = workout.ex.origFrame, workout.ex.moves
            output = cv2.resize(frame, (640, 480))
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (30, 30)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2
            if workout.isStarted:
                if not workout.isRest:

                    # printing current exercise
                    cv2.putText(output, str(workout.currentExercise),
                                    (output.shape[0] // 2, 30),
                                    font,
                                    fontScale,
                                    (253, 35, 91),
                                    4)

                    if workout.isTabata:

                        # workout type
                        cv2.putText(output, 'Tabata',
                                    (output.shape[0] - 10, 30),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
                        
                        # number of moves
                        cv2.putText(output, str(moves) + ' moves',
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        fontScale = 5
                        thickness = 10
                        bottomLeftCornerOfText = (output.shape[0] // 2, output.shape[1] // 2)

                        # time left
                        cv2.putText(output, str(int(workout.thresh - workout.ex.totalTime)),
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    (253, 35, 91),
                                    thickness,lineType)
                    else:
                        # workout type
                        cv2.putText(output, 'Standard',
                                    (output.shape[0] - 10, 30),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        # seconds passed
                        cv2.putText(output, str(int(workout.ex.totalTime)) + ' secs',
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        fontScale = 5
                        thickness = 10
                        bottomLeftCornerOfText = (output.shape[0] // 2, output.shape[1] // 2)

                        # number of moves
                        cv2.putText(output, str(int(workout.thresh - moves)),
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    (253, 35, 91),
                                    thickness,lineType)

                        
                else:

                    # workout type
                    cv2.putText(output, 'Pause',
                                    (output.shape[0] // 2, 30),
                                    font,
                                    fontScale,
                                    (0, 255, 0),
                                    4)

                    fontScale = 5
                    thickness = 10
                    bottomLeftCornerOfText = (output.shape[0] // 2, output.shape[1] // 2)
                    cv2.putText(output, str(int(workout.timeToStart)),
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                (29, 254, 71),
                                thickness,lineType)

            imgencode=cv2.imencode('.jpg',output)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

        except Exception:
            pass
    

def index(request): 
    '''
    Loads the workout page
    '''
    global workout
    workout = Workout()


    global workouts


    workouts = []
    for index, wk in enumerate(Workouts.objects.values_list('workout_name').distinct()):
        workouts.append(wk[0])

    try:
        template = "index.html"
        return render(request,template, {'workouts': workouts})
    except Exception as e:
        print("error", e)


def startWorkout(request):
    global workout, th
    try:
        th.do_run = False
    except:
        pass

    
    workout = Workout()
    try:
        workouts = [workoutName]
    except:
        pass

    for wk in Workouts.objects.values_list('workout_name').distinct():
        try:
            if wk[0] != workoutName:
                workouts.append(wk[0])
        except:
            workouts.append(wk[0])
    training_program, models = {}, {}
    restTimes = []
    isTabata = True
    for ex in exercises:
        training_program[ex.exercise.exercise_name] = ex.numRepeats
        models[ex.exercise.exercise_name] = ex.exercise.model_path
        isTabata = ex.isTabata
        restTimes.append(ex.restTime)

    th = threading.Thread(target = workout.runTraining, args = (training_program, models, isTabata, restTimes))
    th.setDaemon(True)
    th.start()

    th.do_run = True

    try:
        template = "index.html"
        return render(request,template, {'workouts': workouts})
    except Exception as e:
        print("error", e)

def stopWorkout(request):
    try:
        workout.isStarted = False
        th.do_run = False
    except:
        pass

    try:
        workouts = [workoutName]
    except:
        pass

    for wk in Workouts.objects.values_list('workout_name').distinct():
        try:
            if wk[0] != workoutName:
                workouts.append(wk[0])
        except:
            workouts.append(wk[0])
    try:
        template = "index.html"
        return render(request,template, {'workouts': workouts})
    except Exception as e:
        print("error", e)

def showWorkout(request):
    global exercises, workoutName

    workoutName = request.GET['name']
    exercises = Workouts.objects.filter(workout_name=workoutName).order_by('exercise_num')

    template = "displayWorkout.html"

    return render(request,template, {'exercises': exercises})

def playSound(request):

    if workout.playSound:
        workout.playSound = False
        return render(request, 'soundPlay.html')
    elif workout.playSoundFinish:
        workout.playSoundFinish = False
        return render(request, 'soundFinish.html')
    else:
        return render(request, 'blank.html')

def showStats1(request):
    '''
    Updates current live stats while the workout is going.
    '''

    try:

        trainingTime, restingTime, performingTime, totalMoves = workout.training_stats.get('totalTime', 0), workout.training_stats.get('restTime', 0),\
            workout.training_stats.get('exerciseTime', 0), workout.training_stats.get('totalMoves', 0)
    except:
        trainingTime, restingTime, performingTime, totalMoves = 0, 0, 0, 0

    if workout.isFinished:
        return render(request, 'stat1.html', { 'trainingTime': str(int(trainingTime)) + ' s', 
                                                    # 'restingTime': int(restingTime),
                                                    # 'performingTime': int(performingTime), 
                                                    # 'totalMoves': int(totalMoves),
                                                    })
    else:
        return render(request, 'blankMoves.html')

def showStats2(request):
    '''
    Updates current live stats while the workout is going.
    '''

    try:

        trainingTime, restingTime, performingTime, totalMoves = workout.training_stats.get('totalTime', 0), workout.training_stats.get('restTime', 0),\
            workout.training_stats.get('exerciseTime', 0), workout.training_stats.get('totalMoves', 0)
    except:
        trainingTime, restingTime, performingTime, totalMoves = 0, 0, 0, 0

    if workout.isFinished:
        return render(request, 'stat2.html', { 'restingTime': str(int(restingTime)) + ' s',
                                                    })
    else:
        return render(request, 'blankMoves.html')

def showStats3(request):
    '''
    Updates current live stats while the workout is going.
    '''

    try:

        trainingTime, restingTime, performingTime, totalMoves = workout.training_stats.get('totalTime', 0), workout.training_stats.get('restTime', 0),\
            workout.training_stats.get('exerciseTime', 0), workout.training_stats.get('totalMoves', 0)
    except:
        trainingTime, restingTime, performingTime, totalMoves = 0, 0, 0, 0

    if workout.isFinished:
        return render(request, 'stat3.html', { 'performingTime': str(int(performingTime)) + ' s'
                                                    })
    else:
        return render(request, 'blankMoves.html')


def showStats4(request):
    '''
    Updates current live stats while the workout is going.
    '''

    try:

        trainingTime, restingTime, performingTime, totalMoves = workout.training_stats.get('totalTime', 0), workout.training_stats.get('restTime', 0),\
            workout.training_stats.get('exerciseTime', 0), workout.training_stats.get('totalMoves', 0)
    except:
        trainingTime, restingTime, performingTime, totalMoves = 0, 0, 0, 0

    if workout.isFinished:
        return render(request, 'stat4.html', { 'totalMoves': str(int(totalMoves)) + ' rep'
                                                    })
    else:
        return render(request, 'blankMoves.html')

@gzip.gzip_page
def dynamic_stream(request,stream_path="video"):
    '''
    Grabs frame with get frame and streams into the page
    '''
    try :
        return StreamingHttpResponse(get_frame(),content_type="multipart/x-mixed-replace;boundary=frame")
    except :
        return "error"