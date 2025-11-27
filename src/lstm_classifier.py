import os
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from collections import Counter


BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data", "whatsapp_labeled.csv")

# ---------------------
# Dataset PyTorch
# ---------------------
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------
# Modelo LSTM
# ---------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out


# ---------------------
# Preparação
# ---------------------
def tokenize(text_list, max_len=40):
    print("Tokenizando (sem torchtext)...")

    # Lista de listas de tokens
    tokens = [t.split() for t in text_list]

    # Criar vocabulário manualmente
    counter = Counter()
    for sentence in tokens:
        counter.update(sentence)

    # Mapear palavras para índices
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.most_common())}
    vocab["<unk>"] = 0  # token desconhecido

    # Codificar sentenças
    encoded = []
    for sentence in tokens:
        seq = [vocab.get(token, 0) for token in sentence[:max_len]]
        seq += [0] * (max_len - len(seq))
        encoded.append(seq)

    return encoded, vocab



def main():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["texto_limpo"])

    X_text = df["texto_limpo"].astype(str).tolist()

    # Tokenizar
    X_encoded, vocab = tokenize(X_text, max_len=40)

    # Labels
    le = LabelEncoder()
    y = le.fit_transform(df["intent"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, stratify=y
    )

    train_ds = TextDataset(X_train, y_train)
    test_ds = TextDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32)

    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=64,
        hidden_dim=64,
        num_classes=len(le.classes_)
    )

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # Treinar
    for epoch in range(3):
        for X_batch, y_batch in train_dl:
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print("Epoch", epoch+1, "Loss:", float(loss))

    # Avaliar
    correct = total = 0
    for X_batch, y_batch in test_dl:
        preds = model(X_batch)
        predicted = preds.argmax(dim=1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    print("Acurácia da LSTM:", correct / total)


if __name__ == "__main__":
    main()
