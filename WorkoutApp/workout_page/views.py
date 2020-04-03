from django.shortcuts import render, render_to_response
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
            bottomLeftCornerOfText = (20, 20)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2
            if workout.isStarted:
                if not workout.isRest:
                    cv2.putText(output, str(workout.currentExercise),
                                    (output.shape[0] // 2, 20),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
                    if workout.isTabata:
                        cv2.putText(output, 'Tabata',
                                    (output.shape[0] - 20, 20),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        cv2.putText(output, str(moves),
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
                        fontScale = 5
                        thickness = 10
                        bottomLeftCornerOfText = (output.shape[0] // 2, output.shape[1] // 2)
                        cv2.putText(output, str(int(workout.thresh - workout.ex.totalTime)),
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    (0, 0, 255),
                                    thickness,lineType)
                    else:
                        cv2.putText(output, 'Standard',
                                    (output.shape[0] - 20, 20),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        fontScale = 5
                        thickness = 10
                        bottomLeftCornerOfText = (output.shape[0] // 2, output.shape[1] // 2)
                        cv2.putText(output, str(int(workout.thresh - moves)),
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    (0, 0, 255),
                                    thickness,lineType)

                        
                else:
                    fontScale = 5
                    thickness = 10
                    bottomLeftCornerOfText = (output.shape[0] // 2, output.shape[1] // 2)
                    cv2.putText(output, str(int(workout.timeToStart)),
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                (0, 255, 0),
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

    for wk in Workouts.objects.values_list('workout_name').distinct():
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
    workouts = []
    for wk in Workouts.objects.values_list('workout_name').distinct():
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
    workout.isStarted = False
    th.do_run = False


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

def updateStats(request):
    '''
    Updates current live stats while the workout is going.
    '''

    try:
        moves, totalTime, currentExercise = workout.ex.moves, workout.ex.totalTime, workout.currentExercise

        trainingTime, restingTime, performingTime, totalMoves = workout.training_stats.get('totalTime', 0), workout.training_stats.get('restTime', 0),\
            workout.training_stats.get('exerciseTime', 0), workout.training_stats.get('totalMoves', 0)
    except:
        moves, totalTime, currentExercise = 0, 0, 'None'
        trainingTime, restingTime, performingTime, totalMoves = 0, 0, 0, 0


    return render(request, 'liveStats.html', {'moveCounts':int(moves), 
                                                'totalTime': int(totalTime), 
                                                'currentExercise': currentExercise,
                                                'trainingTime': int(trainingTime), 
                                                'restingTime': int(restingTime),
                                                'performingTime': int(performingTime), 
                                                'totalMoves': int(totalMoves),
                                                })

@gzip.gzip_page
def dynamic_stream(request,stream_path="video"):
    '''
    Grabs frame with get frame and streams into the page
    '''
    try :
        return StreamingHttpResponse(get_frame(),content_type="multipart/x-mixed-replace;boundary=frame")
    except :
        return "error"