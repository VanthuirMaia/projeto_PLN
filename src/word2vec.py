import os
import pandas as pd
import spacy
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEAN = os.path.join(BASE_DIR, "data", "whatsapp_chat_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "word2vec.model")

def load_data():
    df = pd.read_csv(DATA_CLEAN)
    df = df.dropna(subset=["texto_limpo"])
    df["texto_limpo"] = df["texto_limpo"].astype(str)
    return df

def tokenize_sentences(df):
    print("Tokenizando texto para Word2Vec...")
    sentences = [row.split() for row in df["texto_limpo"]]
    print(f"Total de sentenças: {len(sentences)}")
    return sentences

def train_word2vec(sentences):
    print("Treinando Word2Vec...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=3,
        workers=4,
        sg=1   # Skip-gram (melhor para palavras raras)
    )
    print("Modelo Word2Vec treinado.")
    return model

def save_model(model):
    model.save(MODEL_PATH)
    print(f"Modelo Word2Vec salvo em: {MODEL_PATH}")


def show_similar_words(model):
    print("\nPalavras mais similares — usando vocabulário do dataset:")

    test_words = ["bhai", "bol", "kore", "class", "college", "exam", "submit", "mail"]

    for word in test_words:
        if word in model.wv:
            print(f"\n'{word}' similares:")
            for similar, score in model.wv.most_similar(word, topn=5):
                print(f"   {similar}  ({score:.3f})")
        else:
            print(f"\nPalavra '{word}' não encontrada no vocabulário.")



def visualize_embeddings(model):
    print("\nReduzindo dimensões com PCA para visualização...")

    words = list(model.wv.index_to_key[:200])  # pegar só 200 palavras
    vectors = model.wv[words]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6)

    for i, word in enumerate(words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]))

    plt.title("Visualização 2D do Word2Vec (PCA)")
    plt.show()

def main():
    df = load_data()
    sentences = tokenize_sentences(df)
    model = train_word2vec(sentences)
    save_model(model)
    show_similar_words(model)
    visualize_embeddings(model)

if __name__ == "__main__":
    main()
