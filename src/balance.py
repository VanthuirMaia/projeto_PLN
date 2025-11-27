import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_LABELED = os.path.join(BASE_DIR, "data", "whatsapp_labeled.csv")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
BALANCED_X = os.path.join(BASE_DIR, "data", "X_balanced.pkl")
BALANCED_Y = os.path.join(BASE_DIR, "data", "y_balanced.pkl")


def load_data():
    df = pd.read_csv(DATA_LABELED)
    df = df.dropna(subset=["texto_limpo"])
    df["texto_limpo"] = df["texto_limpo"].astype(str)
    return df


def vectorize(df):
    vectorizer = joblib.load(VECTORIZER_PATH)
    X = vectorizer.transform(df["texto_limpo"])
    y = df["intent"]
    return X, y


def balance_data(X, y):
    print("Formato original:", X.shape)

    # Passo 1 — Oversample simples para classes muito pequenas (<10)
    ros = RandomOverSampler(sampling_strategy={
        "pergunta": 10,
        "solicitacao": 10,
        "negacao": 23  # manter
    }, random_state=42)

    X_res, y_res = ros.fit_resample(X, y)
    print("Após RandomOverSampler:", X_res.shape)

    # Passo 2 — SMOTE para equilibrar todas as classes
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_final, y_final = smote.fit_resample(X_res, y_res)

    print("Após SMOTE (final):", X_final.shape)
    print("Distribuição final:")
    print(pd.Series(y_final).value_counts())

    return X_final, y_final


def save_outputs(X, y):
    joblib.dump(X, BALANCED_X)
    joblib.dump(y, BALANCED_Y)
    print("\nDados balanceados salvos com sucesso!")


def main():
    df = load_data()
    X, y = vectorize(df)
    Xb, yb = balance_data(X, y)
    save_outputs(Xb, yb)


if __name__ == "__main__":
    main()
