import os
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEAN = os.path.join(BASE_DIR, "data", "whatsapp_chat_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lda_model.gensim")

def load_data():
    df = pd.read_csv(DATA_CLEAN)
    df = df.dropna(subset=["texto_limpo"])
    df["texto_limpo"] = df["texto_limpo"].astype(str)
    return df

def tokenize(df):
    print("Tokenizando para LDA...")
    return [row.split() for row in df["texto_limpo"]]

def prepare_corpus(tokenized_sentences):
    print("Preparando dicionário e corpus...")

    dictionary = corpora.Dictionary(tokenized_sentences)
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    corpus = [dictionary.doc2bow(text) for text in tokenized_sentences]

    return dictionary, corpus

def train_lda(dictionary, corpus, num_topics=5):
    print(f"Treinando LDA com {num_topics} tópicos...")
    
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        random_state=42
    )

    print("Modelo LDA treinado.")
    return lda_model

def show_topics(lda_model):
    print("\nTópicos detectados pelo LDA:\n")
    
    topics = lda_model.print_topics(num_topics=5, num_words=7)
    for idx, topic in topics:
        print(f"Tópico {idx}: {topic}")

def main():
    df = load_data()
    tokenized = tokenize(df)
    dictionary, corpus = prepare_corpus(tokenized)
    lda_model = train_lda(dictionary, corpus)
    show_topics(lda_model)

    # Salvar modelo
    lda_model.save(MODEL_PATH)
    print(f"\nModelo LDA salvo em: {MODEL_PATH}")

if __name__ == "__main__":
    main()
