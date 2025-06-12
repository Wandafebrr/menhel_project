from django.urls import path
from app.views import view_pemodelan as views

urlpatterns = [
    path('pemodelan/', views.lsa_topic_view, name='pemodelan'),
]