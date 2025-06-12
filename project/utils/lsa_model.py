from sklearn.decomposition import TruncatedSVD
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from utils.model_kmeans import clean_text

def generate_best_lsa_topics(docs, vectorizer, min_topics=2, max_topics=10, topn=2):
    if not isinstance(docs, list) or len(docs) < 3:
        return {"error": "Jumlah dokumen terlalu sedikit. Minimal 3 dokumen diperlukan."}

    cleaned_docs = [clean_text(doc) for doc in docs]
    tokenized_docs = [doc.split() for doc in cleaned_docs]

    # Biasanya vectorizer sudah fit, jadi cukup transform
    X_tfidf = vectorizer.transform(cleaned_docs)
    feature_names = vectorizer.get_feature_names_out()

    n_features = X_tfidf.shape[1]
    if n_features == 0:
        return {"error": "Tidak ada fitur setelah filtering TF-IDF."}

    max_topics = min(max_topics, len(docs), n_features)
    if max_topics < min_topics:
        return {"error": "Jumlah topik maksimum kurang dari minimum."}

    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

    best_model = None
    best_num_topics = min_topics
    best_coherence = -1
    best_topics = []

    for n_topics in range(min_topics, max_topics + 1):
        svd_model = TruncatedSVD(n_components=n_topics)
        lsa = svd_model.fit_transform(X_tfidf)

        topics = []
        for topic_weights in svd_model.components_:
            top_keywords_idx = topic_weights.argsort()[::-1][:topn]
            topic_keywords = [feature_names[i] for i in top_keywords_idx]
            topics.append(topic_keywords)

        coherence_model = CoherenceModel(
            topics=topics,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()

        if coherence_score > best_coherence:
            best_num_topics = n_topics
            best_coherence = coherence_score
            best_model = svd_model
            best_topics = topics

    return {
        "best_model": best_model,
        "topics": best_topics,
        "coherence_score": best_coherence,
        "n_topics": best_num_topics
    }
