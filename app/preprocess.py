'''텍스트 정제, 토큰화, 패딩, 라벨 인코딩을 담당하는 모듈'''

from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple, Dict, Sequence, Iterable

import numpy as np
import os

from konlpy.tag import Okt

import matplotlib
matplotlib.use('Agg')

STOPWORDS = {
    '이', '그', '저', '것', '수', '등', '및', '더', '하다', '되다', '있다',
    '없다', '않다', '이다', '하는', '있는', '에', '은', '는', '가', '을',
    '를', '의', '와', '과', '도', '서', '로', '으로', '에서', '까지',
    '부터', '만', '고', '며', '면', '지', '나', '거', '든',
}

def clean_text(text: str, remove_stopwords:bool = True) -> str:
    '''영문 기사 문장에서 특수문자와 불필요한 단어를 제거한다.'''

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^가-힣0-9!?.,' ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# 파일 상단에 전역으로 한 번만 생성
_okt = Okt()

def tokenize(text: str) -> List[str]:
    return _okt.morphs(text, stem=True)  # okt 새로 생성 안 함


def tokenize_all(texts: Sequence[str]) -> List[List[str]]:
    '''모든 텍스트를 미리 토큰화하여 캐싱한다. (JVM 크래시 방지)'''
    print(f"형태소 분석 중... (총 {len(texts)}개)")
    result = []
    for i, text in enumerate(texts):
        tokens = tokenize(clean_text(text))
        tokens = [
            t for t in tokens
            if len(t) >= 2
               and t not in STOPWORDS
               and not t.isdigit()
               and t.isalnum()
        ]
        result.append(tokens)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(texts)} 완료")
    print("형태소 분석 완료!")
    return result


def build_vocab(texts: Sequence[str], max_vocab: int) -> Dict[str, int]:
    '''학습 데이터에서 자주 등장한 단어를 정수 인덱스로 매핑하는 사전을 만든다.'''
    tokenized = tokenize_all(texts)
    return build_vocab_from_tokens(tokenized, max_vocab)


def build_vocab_from_tokens(tokenized_texts: Sequence[List[str]], max_vocab: int) -> Dict[str, int]:
    '''이미 토큰화된 결과로 vocab을 만든다.'''
    counter: Counter[str] = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    most_common = counter.most_common(max_vocab - 2)
    vocab = {"<PAD>": 0, "<OOV>": 1}
    for index, (word, _) in enumerate(most_common, start=2):
        vocab[word] = index

    # ── Word Frequency Histogram ────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    font_prop = fm.FontProperties(fname='temp/NanumGothic.ttf')

    top30_words = [word for word, _ in most_common[:30]]
    top30_counts = [count for _, count in most_common[:30]]

    os.makedirs('./result', exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.bar(top30_words, top30_counts, color='steelblue')
    plt.title(f'Top 30 Word Frequencies (Total vocab: {len(vocab)})', fontsize=14,
              fontproperties=font_prop)
    plt.xlabel('Word', fontproperties=font_prop)
    plt.ylabel('Frequency', fontproperties=font_prop)
    plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig('./result/vocab_histogram.png', dpi=150)
    plt.close()
    print("vocab_histogram.png saved")
    # ────────────────────────────────────────────────────────────

    return vocab

def texts_to_sequences(tokenized_texts: Sequence[List[str]], vocab: Dict[str, int]) -> List[List[int]]:
    '''토큰화된 문장 목록을 정수 토큰 시퀀스 목록으로 변환한다.'''
    sequences: List[List[int]] = []
    for tokens in tokenized_texts:
        seq = [vocab.get(token, vocab["<OOV>"]) for token in tokens]
        sequences.append(seq)
    return sequences

def pad_sequences(sequences: Sequence[Sequence[int]], max_len: int) -> np.ndarray:
    '''서로 다른 길이의 정수 시퀀스를 동일한 길이의 2차원 배열로 맞춘다.'''

    padded = np.zeros((len(sequences), max_len), dtype=np.int64)  # 모든 값을 0으로 채운 패딩 배열을 먼저 만든다.
    for i, seq in enumerate(sequences):  # 각 시퀀스와 해당 위치를 함께 순회한다.
        # 각 토큰이 STOP_WORDS에 포함되지 않는 경우만 남기고, 뒤쪽 기준으로 max_len만큼 자른다.
        truncated = [token for token in seq][-max_len:]

        # 짧은 문장은 앞쪽을 0으로 남기고 뒤쪽에 토큰을 채운다.
        if truncated:
            padded[i, -len(truncated):] = truncated

    return padded  # 패딩이 끝난 2차원 배열을 반환한다.

def encode_labels(labels:Sequence[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    '''문자열 라벨을 정수 라벨로 변환하고 양방향 라벨 사전을 반환한다.'''

    label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))} # 라벨명을 정수 ID로 매핑한다.
    id_to_label = {idx: label for label, idx in label_to_id.items()} # 예측 결과 해석을 위해 정수 ID를 라벨명으로 되돌리는 사전을 만든다.
    encoded = np.array([label_to_id[label] for label in labels], dtype=np.int64) # 각 정답 라벨을 정수로 변환한다.
    return encoded, label_to_id, id_to_label # 인코딩 결과와 라벨 사전들을 반환한다.