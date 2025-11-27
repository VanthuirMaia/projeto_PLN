import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEAN = os.path.join(BASE_DIR, "data", "whatsapp_chat_clean.csv")


def load_data():
    df = pd.read_csv(DATA_CLEAN)
    df = df.dropna(subset=["texto_limpo"])
    df["texto_limpo"] = df["texto_limpo"].astype(str)
    return df


def build_retriever(texts):
    print("Carregando modelo de embeddings (SentenceTransformer)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Gerando embeddings das mensagens...")
    embeddings = embedder.encode(texts, show_progress_bar=True)

    print("Construindo índice de vizinhos mais próximos (cosine)...")
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(embeddings)

    return embedder, nn, embeddings


def load_generator():
    print("Carregando modelo generativo (FLAN-T5-small)...")
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def answer_question(query, texts, embedder, nn, tokenizer, model):
    # 1) Embedding da pergunta
    q_emb = embedder.encode([query])

    # 2) Recuperar mensagens mais próximas
    distances, indices = nn.kneighbors(q_emb, n_neighbors=5)
    retrieved = [texts[i] for i in indices[0]]

    contexto = "\n".join(retrieved)

    # 3) Geração da resposta condicionada ao contexto
    prompt = (
    "Use APENAS as mensagens abaixo como base para responder.\n"
    "Se o contexto não fornecer informação relevante, diga:\n"
    "'Não há informações suficientes nas mensagens recuperadas.'\n\n"
    "Mensagens recuperadas:\n"
    f"{contexto}\n\n"
    f"Pergunta do usuário: {query}\n"
    "Explique em português claro."
    )


    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = model.generate(**inputs, max_new_tokens=128)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer, retrieved


def main():
    df = load_data()
    texts = df["texto_limpo"].tolist()

    embedder, nn, embeddings = build_retriever(texts)
    tokenizer, model = load_generator()

    # Exemplo de pergunta
    query = "O que estão falando sobre PDF?"

    answer, retrieved = answer_question(
        query, texts, embedder, nn, tokenizer, model
    )

    print("\nPergunta:")
    print(query)

    print("\nMensagens recuperadas:")
    for m in retrieved:
        print("-", m)

    print("\nResposta gerada:")
    print(answer)


if __name__ == "__main__":
    main()
