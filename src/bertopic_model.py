import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data", "whatsapp_chat_clean.csv")

def load_data():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["texto_limpo"])
    df["texto_limpo"] = df["texto_limpo"].astype(str)
    return df

def main():
    df = load_data()
    texts = df["texto_limpo"].tolist()

    print("Carregando modelo Transformer...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Gerando tópicos com BERTopic...")
    topic_model = BERTopic(embedding_model=embedder, language="multilingual")
    topics, probs = topic_model.fit_transform(texts)

    print("\nTópicos detectados:")
    print(topic_model.get_topic_info().head())

    print("\nPalavras do Tópico 0:")
    print(topic_model.get_topic(0))

    topic_model.save("models/bertopic_model")

if __name__ == "__main__":
    main()
