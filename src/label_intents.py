import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEAN = os.path.join(BASE_DIR, "data", "whatsapp_chat_clean.csv")
DATA_LABELED = os.path.join(BASE_DIR, "data", "whatsapp_labeled.csv")

def load_data():
    df = pd.read_csv(DATA_CLEAN)
    print(f"Dataset carregado: {df.shape[0]} mensagens")
    return df

def detect_intent(text):
    text = text.lower().strip()

    # ---------------------------------------
    # SAUDAÇÃO
    # ---------------------------------------
    saudacoes = ["oi", "olá", "ola", "hey", "hi", "hello", "e aí", "eai"]
    if any(text.startswith(s) for s in saudacoes):
        return "saudacao"

    # ---------------------------------------
    # PERGUNTA
    # ---------------------------------------
    perguntas_inicio = [
        "por que", "pq", "como", "onde", "quando", "qual",
        "can", "could", "do you", "did you", "are you", "is it"
    ]
    if "?" in text or any(text.startswith(p) for p in perguntas_inicio):
        return "pergunta"

    # ---------------------------------------
    # CONFIRMAÇÃO
    # ---------------------------------------
    confirmacoes = ["sim", "ok", "certo", "beleza", "claro", "yes", "yeah", "yup", "sure", "indeed"]
    if text in confirmacoes:
        return "confirmacao"

    # ---------------------------------------
    # NEGAÇÃO
    # ---------------------------------------
    negacoes = ["não", "nao", "no", "nope", "never"]
    if any(text.startswith(n) for n in negacoes):
        return "negacao"

    # ---------------------------------------
    # AGRADECIMENTO
    # ---------------------------------------
    agradecimentos = ["obrigado", "obg", "vlw", "valeu", "thanks", "thank", "thx", "ty"]
    if any(a in text for a in agradecimentos):
        return "agradecimento"

    # ---------------------------------------
    # SOLICITAÇÃO (NOVA!)
    # ---------------------------------------
    solicitacao = [
        "please", "can you", "could you", "me envia", "manda", "mande",
        "pode enviar", "pode mandar", "pode fazer", "send me", "give me"
    ]
    if any(text.startswith(s) for s in solicitacao):
        return "solicitacao"

    # ---------------------------------------
    # DEFAULT
    # ---------------------------------------
    return "outro"


def apply_intents(df):
    print("Gerando rótulos de intenção...")
    df["intent"] = df["texto_limpo"].astype(str).apply(detect_intent)
    
    print(df["intent"].value_counts())
    return df

def save_labeled(df):
    df.to_csv(DATA_LABELED, index=False)
    print(f"\nArquivo salvo com rótulos em:\n{DATA_LABELED}")

def main():
    df = load_data()
    df = apply_intents(df)
    save_labeled(df)

if __name__ == "__main__":
    main()
