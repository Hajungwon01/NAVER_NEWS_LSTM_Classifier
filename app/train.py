"""데이터 전처리, 모델 학습, 평가, 저장을 수행하는 모듈."""

from __future__ import annotations

import os
import pickle
import random
from typing import Dict, Tuple


import numpy as np
import torch

torch.set_num_threads(1)  # 작은 실습 데이터에서는 CPU 스레드를 1개로 제한하여 실행 환경별 지연을 줄인다.
torch.backends.mkldnn.enabled = False  # 일부 CPU 환경에서 LSTM 연산이 오래 멈추는 문제를 피하기 위해 MKLDNN을 비활성화한다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import Config
from app.data import load_sample_data
from app.model import TextLSTMClassifier
from app.preprocess import build_vocab_from_tokens, clean_text, encode_labels, pad_sequences, texts_to_sequences, tokenize_all

import matplotlib
matplotlib.use('Agg')

def set_seed(seed: int) -> None:
    """학습 결과가 최대한 동일하게 재현되도록 난수를 고정한다."""

    random.seed(seed)              # 파이썬 random 모듈의 난수를 고정한다.
    np.random.seed(seed)           # NumPy 난수를 고정한다.
    torch.manual_seed(seed)        # PyTorch CPU 난수를 고정한다.

def train_model(config: Config) -> Tuple[TextLSTMClassifier, Dict[str, object]]:
    """샘플 BBC 기사 데이터를 사용해 LSTM 문서 분류 모델을 학습한다."""

    set_seed(config.random_state)
    raw_texts, labels = load_sample_data()
    cleaned_texts = [clean_text(text) for text in raw_texts]
    tokenized_texts = tokenize_all(cleaned_texts)  # 미리 토큰화
    vocab = build_vocab_from_tokens(tokenized_texts, config.max_vocab)
    sequences = texts_to_sequences(tokenized_texts, vocab)
    x = pad_sequences(sequences, config.max_len)
    y, label_to_id, id_to_label = encode_labels(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_dataset  = TensorDataset(torch.tensor(x_test),  torch.tensor(y_test))
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=config.batch_size)

    model = TextLSTMClassifier(
        vocab_size=len(vocab), embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim, num_classes=len(label_to_id)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # ── 기록용 리스트 ────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": []}
    # ────────────────────────────────────────────────────────────

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss   = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (torch.argmax(logits, dim=1) == batch_y).sum().item()
            total      += len(batch_y)

        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        print(f"Epoch {epoch:02d}/{config.epochs} "
              f"- train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f}")

    # ── 시각화 1: Train Loss ─────────────────────────────────────
    import matplotlib.pyplot as plt

    epochs = range(1, config.epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], marker='o', label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./result/loss_curve.png', dpi=150)
    plt.close()


    # ── 시각화 2: Train Accuracy ─────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_acc"], marker='o', label='Train Accuracy')
    plt.title('Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./result/accuracy_curve.png', dpi=150)
    plt.close()

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.tolist())
            all_targets.extend(batch_y.tolist())

    target_names = [id_to_label[i] for i in range(len(id_to_label))]

    # ── 평가 지표 ────────────────────────────────────────────────────
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print()
    print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))
    # ────────────────────────────────────────────────────────────────

    # ── Confusion Matrix 시각화 ──────────────────────────────────────
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import matplotlib.font_manager as fm

    font_prop = fm.FontProperties(fname='temp/NanumGothic.ttf')

    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)

    # set_xticks/set_yticks에서 fontproperties 제거
    ax.set_xticks(range(len(target_names)))  # fontproperties 없이
    ax.set_yticks(range(len(target_names)))  # fontproperties 없이
    ax.set_xticklabels(target_names, rotation=45, ha='right', fontproperties=font_prop)
    ax.set_yticklabels(target_names, fontproperties=font_prop)
    ax.set_xlabel('Predicted', fontproperties=font_prop)
    ax.set_ylabel('Actual', fontproperties=font_prop)
    ax.set_title('Confusion Matrix', fontproperties=font_prop)
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig('./result/confusion_matrix.png', dpi=150)
    plt.close()


    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    torch.save(model.state_dict(), config.model_path)
    with open(config.model_path.replace(".pt", "_meta.pkl"), "wb") as f:
        pickle.dump({
            "vocab": vocab, "label_to_id": label_to_id,
            "id_to_label": id_to_label, "config": config
        }, f)

    metadata = {
        "vocab": vocab, "label_to_id": label_to_id,
        "id_to_label": id_to_label, "accuracy": accuracy,
        "history": history
    }
    return model, metadata                             # 학습된 모델과 메타데이터를 반환한다.