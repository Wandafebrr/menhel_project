from django.db import models
from .model_user import User
from .model_chat_session import ChatSession

class ChatLogs(models.Model):
    # user = models.ForeignKey(User, related_name='chat', on_delete=models.SET_NULL, null=True)
    message = models.CharField(null=True)
    response = models.CharField(null=True)
    tanggal = models.DateTimeField(auto_now_add=True, null=True)
    session = models.ForeignKey(ChatSession, related_name='chat_logs', on_delete=models.SET_NULL, null=True)
    score = models.IntegerField(null=True, default=0)