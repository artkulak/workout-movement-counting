from django.urls import path, re_path
from django.contrib import admin
from django.conf.urls import url
from . import views

urlpatterns = [
    # url for grabbing frames while performing workout
    re_path(r'^/(?P<stream_path>(.*?))/$', views.dynamic_stream,name="videostream"),  
    path('',views.index, name='index'),

    # urls for showing stats after workout finished
    path('showStats1/', views.showStats1, name="showStats1"),
    path('showStats2/', views.showStats2, name="showStats2"),
    path('showStats3/', views.showStats3, name="showStats3"),
    path('showStats4/', views.showStats4, name="showStats4"),

    path('startWorkout/', views.startWorkout, name="startWorkout"),
    path('stopWorkout/', views.stopWorkout, name="stopWorkout"),
    path('showWorkout/', views.showWorkout, name="showWorkout"),
    path('playSound/', views.playSound, name="playSound"),


    # urls to redirect for workout adding
    path('goAdmin/', admin.site.login, name='goAdmin')
]
