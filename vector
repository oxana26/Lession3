import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

# Пример базы знаний (документы)
documents = [
    "Как сбросить пароль в системе?",
    "Где найти настройки VPN?",
    "Как установить обновление?",
    "Почему не работает интернет?",
    "Как создать новую учетную запись?"
]

# 1. Векторизация с помощью TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(documents).toarray()

# 2. Векторизация с помощью BERT
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
bert_vectors = model.encode(documents)

# 3. Создание FAISS-индекса для BERT-векторов
dim = bert_vectors.shape[1]
index = faiss.IndexFlatIP(dim)  # Используем косинусное сходство
index.add(bert_vectors.astype('float32'))
