import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "whatsapp_chat_clean.csv")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "tfidf_features.pkl")

def load_clean_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Arquivo limpo não encontrado em {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print("Dados limpos carregados.")
    return df

def apply_tfidf(df):
    print("Criando vetorizer TF-IDF...")

    # --- CORREÇÃO DOS NANs ---
    df = df.dropna(subset=["texto_limpo"])
    df["texto_limpo"] = df["texto_limpo"].astype(str)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        min_df=3
    )

    X = vectorizer.fit_transform(df["texto_limpo"])
    print("TF-IDF aplicado, forma da matriz:", X.shape)

    return X, vectorizer


def save_outputs(X, vectorizer):
    # criar pasta models caso não exista
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("Salvando vetorizer e features...")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(X, FEATURES_PATH)

    print("\nVetorização salva com sucesso:")
    print(f" - Vetorizer: {VECTORIZER_PATH}")
    print(f" - TF-IDF matrix: {FEATURES_PATH}")

def main():
    df = load_clean_data()
    X, vectorizer = apply_tfidf(df)
    save_outputs(X, vectorizer)

if __name__ == "__main__":
    main()
