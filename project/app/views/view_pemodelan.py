from django.shortcuts import render
import joblib
from utils.model_kmeans import clean_text
import os
from django.conf import settings
from utils.lsa_model import generate_best_lsa_topics


# Load model & vectorizer
vectorizer = joblib.load(settings.MODEL_DIR / 'vectorizer.pkl')
scaler = joblib.load(settings.MODEL_DIR / 'scaler.pkl')
svd_model = joblib.load(settings.MODEL_DIR / 'svd_model.pkl')
kmeans_model = joblib.load(settings.MODEL_DIR / 'kmeans_model.pkl')
cluster_labels = joblib.load(settings.MODEL_DIR / 'cluster_labels.pkl')


def predict_cluster(text_input):
    cleaned = clean_text(text_input)
    text_vec = vectorizer.transform([cleaned])
    text_scaled = scaler.transform(text_vec)
    text_reduced = svd_model.transform(text_scaled)
    cluster_number = kmeans_model.predict(text_reduced)[0]
    cluster_label = cluster_labels.get(cluster_number, "Unknown Cluster")
    return cluster_label


def lsa_topic_view(request):
    context = {}

    if request.method == 'POST':
        # Ambil teks input dari textarea
        text_input = request.POST.get('text_input', '').strip()

        if text_input:
            prediction = predict_cluster(text_input)
            context["prediction"] = prediction

        # Untuk topik modeling, ambil seluruh teks dari dokumen (bisa juga text_input jika single input)
        docs_text = request.POST.get('documents', text_input)  # fallback ke text_input kalau form 1 saja
        docs = [doc.strip() for doc in docs_text.splitlines() if doc.strip()]

        # Jalankan LSA jika ada input valid
        if docs:
            result = generate_best_lsa_topics(
                docs=docs,
                vectorizer=vectorizer,
                min_topics=2,
                max_topics=10,
                topn=4
            )

            if "error" in result:
                context["topic_list"] = "Gagal menghasilkan topik."
            else:
                topics = result["topics"]
                topic_list = [", ".join(t) for t in topics]
                joined_topics = " | ".join(topic_list)
                context["topic_list"] = joined_topics

    return render(request, 'pemodelan/index.html', context)
