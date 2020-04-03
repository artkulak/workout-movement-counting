from django.contrib import admin

# Register your models here.

from .models import Exercise, Workouts

admin.site.register(Exercise)
admin.site.register(Workouts)


