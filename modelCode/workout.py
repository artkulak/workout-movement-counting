from Inference import ExerciseCapture
from time import time
from time import sleep

from warnings import filterwarnings
filterwarnings('ignore')

# current training program
training_program = {
    'push-up': 10,
    'squat': 10,
    'sit-up': 10
}

# models for each movement prediction
models = {
    'push-up': 'model.pt',
    'squat': 'model.pt',
    'sit-up': 'model.pt'
}


def runTraining(training_program, models, tabata=False, restTime=5):
    '''
    Performs move counting and stats collection for each exercise while training. Can perform tabata or just simple move counting
    :param training_program: The program on which training is performed
    :param models: models for movement counting for each exercise
    :param tabata: If the training should be tabata or not
    :param restTime: time to rest between each exercise
    :return: Collected stats for each training
    '''
    timeStart = time()
    timeEx = 0
    totalMoves = 0

    training_stats = {}
    for index, (key, value) in enumerate(training_program.items()):

        model = models[key]
        if not tabata:
            print(f'Performing {value} {key}')
        else:
            print(f'Performing {key} for {value} seconds')
        ex = ExerciseCapture(model, fromStream=True, timeWise=tabata, thresh=value)
        moves, totalTime = ex.runPipeline()

        timeEx += totalTime
        totalMoves += moves

        training_stats[key+f'_{str(index)}'] = {'moves': moves, 'time': totalTime}
        print(f'Exercise {key} finishes; Moves {moves} Time {totalTime}')

        print(f'Rest for {restTime} seconds')
        sleep(restTime)

    trainingTime = time() - timeStart
    restingTime = trainingTime - timeEx
    print(f'Training has taken {trainingTime}. You have been resting for {restingTime} and performing for {timeEx} seconds \n Total moves {totalMoves}')

    training_stats['totalTime'] = trainingTime
    training_stats['restTime'] = restingTime
    training_stats['exerciseTime'] = timeEx
    training_stats['totalMoves'] = totalMoves

    return training_stats


if __name__ == '__main__':
    stats = runTraining(training_program, models, tabata = True, restTime = 20)
    print(stats)