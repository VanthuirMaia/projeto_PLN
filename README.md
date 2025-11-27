# 🧠 Projeto de Processamento de Linguagem Natural

### Análise de Intenções, Modelagem de Tópicos e RAG em Conversas de WhatsApp

Este repositório contém o desenvolvimento completo do projeto final da disciplina **Processamento de Linguagem Natural**, da **Residência em IA Generativa – UPE**.

O foco foi aplicar técnicas clássicas e modernas de PLN em um **dataset real de conversas de WhatsApp**, explorando desde TF-IDF até Transformers e RAG.

---

## 📌 Objetivos

- Aplicar o pipeline completo de PLN visto em aula.
- Limpar e pré-processar um corpus real (WhatsApp).
- Extrair intenções das mensagens.
- Comparar técnicas clássicas e modernas:
  - TF-IDF
  - Word2Vec
  - LDA
  - LSTM
  - BERTopic (Transformers)
  - RAG (Retrieval-Augmented Generation)
- Demonstrar a evolução histórica das abordagens.

---

## 📁 Estrutura do Projeto

```
projeto_PLN/
│
├── data/
│   ├── whatsapp_chat_raw.csv
│   ├── whatsapp_chat_clean.csv
│   ├── tfidf_features.pkl
│   └── whatsapp_labeled.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── intent_classifier_balanced.pkl
│   ├── word2vec.model
│   ├── lda_model.gensim
│   └── bertopic_model/
│
├── src/
│   ├── preprocess.py
│   ├── vectorize.py
│   ├── label_intents.py
│   ├── balance.py
│   ├── classify.py
│   ├── classify_balanced.py
│   ├── word2vec.py
│   ├── lda_topics.py
│   ├── bertopic_model.py
│   └── rag_demo.py
│
└── README.md
```

---

## 🔧 Tecnologias Utilizadas

- Python 3.12
- spaCy
- scikit-learn
- gensim
- PyTorch
- BERTopic
- SentenceTransformers
- FAISS / sklearn NearestNeighbors
- HuggingFace Transformers

---

## 🧹 1. Pré-processamento (spaCy)

- Normalização
- Remoção de pontuação
- Lemmatização
- Conversão para minúsculas
- Remoção de mensagens inválidas

Saída: `whatsapp_chat_clean.csv`

---

## 🧾 2. Extração de Características

### ✔️ TF-IDF

Usado para o classificador clássico.

Arquivos gerados:

- `tfidf_vectorizer.pkl`
- `tfidf_features.pkl`

---

## 🏷️ 3. Rotulagem Automática de Intenções

Categorias utilizadas:

- saudacao
- confirmacao
- negacao
- agradecimento
- pergunta
- solicitacao
- outro

Saída: `whatsapp_labeled.csv`

---

## ⚖️ 4. Balanceamento (SMOTE)

Aplicado para corrigir o desbalanceamento do dataset.  
Resultado: todas as classes com **10.366 instâncias**.

---

## 🤖 5. Classificação de Intenções

Modelo: **Logistic Regression + TF-IDF**  
Acurácia (dados balanceados): **98%**

Modelo salvo:

- `intent_classifier_balanced.pkl`

---

## 🧬 6. Word Embeddings — Word2Vec

Treinado diretamente no corpus.  
Permitindo análise semântica e visualização com PCA.

Modelo salvo:

- `word2vec.model`

---

## 📚 7. Modelagem de Tópicos

### ✔️ LDA

Gerado com `gensim` → 5 tópicos.

### ✔️ BERTopic

Utilizando Transformers + UMAP + HDBSCAN.  
Melhor desempenho nos tópicos.

---

## 🔍 8. RAG — Retrieval-Augmented Generation

Pipeline implementado:

- Embeddings com SentenceTransformer
- Recuperação com NearestNeighbors
- Geração com FLAN-T5-small

Permite responder perguntas sobre o dataset.

---

## 📎 Fonte do Dataset

WhatsApp Chat Dataset — Kaggle  
https://www.kaggle.com/datasets/rijudhara/whatsappchat

---

## 👤 Autor

Vanthuir Maia  
Residência em IA Generativa — UPE  
2025
