import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Caminhos dos arquivos balanceados
BALANCED_X = os.path.join(BASE_DIR, "data", "X_balanced.pkl")
BALANCED_Y = os.path.join(BASE_DIR, "data", "y_balanced.pkl")

def load_balanced_data():
    X = joblib.load(BALANCED_X)
    y = joblib.load(BALANCED_Y)
    print(f"Dados balanceados carregados: {X.shape}, rótulos: {len(y)}")
    return X, y

def train_model(X_train, y_train):
    print("\nTreinando modelo Logistic Regression com dados balanceados...")
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso.")
    return model

def evaluate(model, X_test, y_test):
    print("\nAvaliação do modelo balanceado:")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.title("Matriz de Confusão - Modelo Balanceado")
    plt.show()

def main():
    X, y = load_balanced_data()

    # Dividir treino/teste com estratificação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)

    # Salvar modelo final
    model_path = os.path.join(BASE_DIR, "models", "intent_classifier_balanced.pkl")
    joblib.dump(model, model_path)
    print(f"\nModelo balanceado salvo em: {model_path}")

if __name__ == "__main__":
    main()
