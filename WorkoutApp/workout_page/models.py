from django.db import models

class Exercise(models.Model):
    '''
    Table in the database for a single exercise
    '''
    exercise_name = models.CharField(max_length = 100)
    exercise_desc = models.TextField()

    # ml model path to handle the exercise
    model_path = models.CharField(max_length = 100)

    def __str__(self):
        return self.exercise_name

class Workouts(models.Model):
    '''
    Table in the database for the workout, which consists of several
    exercises
    '''
    workout_name = models.CharField(max_length = 100)
    
    exercise_num = models.IntegerField(default = 1)

    # link to the Exercise table
    exercise = models.ForeignKey(Exercise, on_delete=models.CASCADE)

    # num repeats/timing
    numRepeats = models.IntegerField(default = 10)

    # rest time after exercise
    restTime = models.IntegerField(default = 10)

    # is the exercise a tabata one
    isTabata = models.BooleanField(default = False)

    def __str__(self):
        return self.workout_name
