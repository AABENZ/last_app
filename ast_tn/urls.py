from django.urls import path
from django.urls.conf import include
from . import views

urlpatterns = [
    # path("test/", views.getData ),
    path("speech_recognition/", views.SpeechRecognitionView.as_view()),
    # path("long_audio/", views.long_audio.as_view()),
]

