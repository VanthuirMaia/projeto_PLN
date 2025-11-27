import os
import pandas as pd
import spacy

# Caminho base do projeto (pasta raiz)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "whatsapp_chat.csv")

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Carrega o CSV de chat do WhatsApp.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado em: {path}")
    
    df = pd.read_csv(path)
    print("Arquivo carregado com sucesso!")
    print("Colunas encontradas:", df.columns.tolist())
    print("\nPrimeiras linhas:")
    print(df.head())
    return df

def init_spacy():
    """
    Carrega o modelo de português do spaCy.
    """
    print("Carregando modelo spaCy pt_core_news_sm...")
    nlp = spacy.load("pt_core_news_sm")
    print("Modelo carregado.")
    return nlp

def basic_cleaning(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Faz uma limpeza básica:
    - converte para string
    - aplica spaCy
    - remove stopwords
    - pega lemas (formas canônicas das palavras)
    
    Retorna o DataFrame com uma nova coluna: 'texto_limpo'
    """
    nlp = init_spacy()

    def process_text(text: str) -> str:
        doc = nlp(str(text))
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and token.is_alpha
        ]
        return " ".join(tokens)

    print(f"\nLimpando textos da coluna: {text_column!r} ...")
    df["texto_limpo"] = df[text_column].astype(str).apply(process_text)
    print("Limpeza concluída. Exemplo:")
    print(df[["texto_limpo"]].head())
    return df

def main():
    # 1) Carregar dados
    df = load_data()

    # 2) Escolher a coluna de texto do dataset
    # 👉 Aqui a gente ainda não sabe o nome exato, então vamos só imprimir
    #    as colunas e você olha qual parece ser a mensagem (ex: 'message', 'text' etc.)
    print("\nAgora você precisa olhar as colunas acima e escolher QUAL é a coluna de texto.")
    print("Sugestão: provavelmente algo como 'message', 'text' ou parecido.")

    # Por enquanto, vou colocar um nome genérico.
    # Assim que você souber o nome real, trocamos aqui.
    text_col = "message"  # <-- TROCAR depois se o nome for outro

    if text_col not in df.columns:
        raise ValueError(
            f"A coluna '{text_col}' não existe no CSV. "
            "Olhe a lista de colunas impressa acima e me diga qual é o nome correto."
        )

    # 3) Limpeza básica de texto
    df = basic_cleaning(df, text_column=text_col)

    # 4) Salvar uma versão processada (opcional, mas útil)
    output_path = os.path.join(BASE_DIR, "data", "whatsapp_chat_clean.csv")
    df.to_csv(output_path, index=False)
    print(f"\nArquivo com texto limpo salvo em: {output_path}")

if __name__ == "__main__":
    main()
