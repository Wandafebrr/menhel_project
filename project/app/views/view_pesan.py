from app.models import ChatLogs, User, ChatSession
from django.db import models
from django.db.models import Q
from django.shortcuts import render, get_object_or_404
import requests
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from datetime import datetime
from django.views import View
from django.views.decorators.http import require_POST

def pesan(request):
    return render(request, 'pesan/index.html')

@csrf_exempt
def chat_with_rasa(request):
    if request.method == "POST":
        data = json.loads(request.body)
        message = data.get("message")
        session = ChatSession.objects.filter(user=request.user, is_open=True).first()

        chat_logs = ChatLogs.objects.create(
            session=session,
            message=message,
            tanggal=datetime.now()
        )
        
        rasa_url = "http://localhost:5005/webhooks/rest/webhook"
        payload = {"sender": "user", "message": message}
        response = requests.post(rasa_url, json=payload)
        
        if response.status_code == 200:
            rasa_response = response.json()
            reply = rasa_response[0].get("text") if rasa_response else "Maaf, tidak ada balasan."
            chat_logs.response = reply
            chat_logs.save()
            return JsonResponse({"reply": reply})
        else:
            return JsonResponse({"error": "Gagal menghubungi Rasa"}, status=500)
    return JsonResponse({"error": "Invalid method"}, status=405)

def get_chat_logs(request):
    logs = ChatLogs.objects.filter(session__user = request.user).order_by('tanggal')
    data = [{
        "id": log.id,
        "message": log.message,
        "response": log.response,
        "datetime": log.tanggal.strftime('%I:%M %p'),
        "score": log.score if log.score is not None else 0
    } for log in logs]
    return JsonResponse({"chat_logs": data})

@csrf_exempt
def create_chat_session(request):
    if request.method == "POST":
        jumlah_session = ChatSession.objects.filter(user=request.user).count()
        user = request.user
        session = ChatSession.objects.create(user=user, nama=f"sesi {jumlah_session + 1}")
        return JsonResponse({"session_id": session.id})
    return JsonResponse({"error": "Invalid method"}, status=405)

def get_chat_session_status(request):
    open_session = ChatSession.objects.filter(user=request.user, is_open=True).first()
    if open_session:
        return JsonResponse({
            "session_exists": True,
        })
    return JsonResponse({"session_exists": False})

@require_POST
@csrf_exempt
def close_chat_session(request):
    session = ChatSession.objects.filter(user=request.user, is_open=True).first()
    if session:
        session.is_open = False
        session.save()
        return JsonResponse({"status": "closed"})
    return JsonResponse({"error": "No open session"}, status=404)

def update_score(request):
    if request.method == "POST":
        data = json.loads(request.body)
        chatlog_id = data.get("chatlog_id")
        score = data.get("score")
        
        chatlog = get_object_or_404(ChatLogs, id=chatlog_id)
        chatlog.score = score
        chatlog.save()

        return JsonResponse({"status": "success", "score": score})
    
    return JsonResponse({"error": "Invalid method"}, status=405)