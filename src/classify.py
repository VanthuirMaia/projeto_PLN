import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_LABELED = os.path.join(BASE_DIR, "data", "whatsapp_labeled.csv")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

def load_data():
    df = pd.read_csv(DATA_LABELED)
    print("Dados com rótulos carregados:", df.shape)
    return df

def load_vectorizer():
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Vetorizer carregado.")
    return vectorizer

def vectorize_texts(vectorizer, texts):
    return vectorizer.transform(texts)

def train_model(X_train, y_train):
    print("\nTreinando modelo Logistic Regression...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso.")
    return model

def evaluate(model, X_test, y_test):
    print("\nAvaliação do modelo:")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.title("Matriz de Confusão - Intenções")
    plt.show()

def main():
    # Carregar dados e vetorizer
    df = load_data()
    vectorizer = load_vectorizer()

    # LIMPEZA: remover NaN e garantir string
    df = df.dropna(subset=["texto_limpo"])
    df["texto_limpo"] = df["texto_limpo"].astype(str)

    # Separar textos e rótulos
    X = vectorize_texts(vectorizer, df["texto_limpo"])
    y = df["intent"]

    # Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Treinar modelo
    model = train_model(X_train, y_train)

    # Avaliar
    evaluate(model, X_test, y_test)

    # Salvar modelo
    model_path = os.path.join(BASE_DIR, "models", "intent_classifier.pkl")
    joblib.dump(model, model_path)
    print(f"\nModelo salvo em: {model_path}")

if __name__ == "__main__":
    main()
