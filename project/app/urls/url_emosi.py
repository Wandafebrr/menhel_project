from django.urls import path
from app.views import view_emosi as views

urlpatterns = [
    path('emosi', views.emosi, name='emosi'),
    path('chat_list', views.chat_list, name='chat_list'),
]