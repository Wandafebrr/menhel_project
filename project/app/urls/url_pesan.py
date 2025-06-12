from django.urls import path
from app.views import view_pesan as views

urlpatterns = [
    path('pesan', views.pesan, name='pesan'),
    path('chat/', views.chat_with_rasa, name='chat_with_rasa'),
    path("api/chat-logs/", views.get_chat_logs, name="get_chat_logs"),
    path("create-chat-session/", views.create_chat_session, name="create_chat_session"),
    path("chat-session-status/", views.get_chat_session_status, name="get_chat_session_status"),
    path("close-chat-session/", views.close_chat_session, name="close_chat_session"),
    path("update-score/", views.update_score, name="update_score"),
]