"""
한국어 웹 텍스트(블로그·뉴스·커뮤니티 등 일반 말글)로 학습 데이터(data/input.txt)를 만드는 스크립트.

- 출처: HuggingFace HAERAE-HUB/KOREAN-WEBTEXT (cc100, OSCAR 등 웹 크롤링 코퍼스를 필터링·중복 제거한 것).
  데이터셋에 라이선스가 명시돼 있지 않으므로 학습·연구 용도로만 쓰고, 데이터 파일은 저장소에 올리지 않음.
- 위키백과는 각주·참고문헌·표 같은 비(非)산문 텍스트가 많아 일반 문장 학습에 맞지 않아 웹 텍스트로 교체함.
- 첫 번째 조각(약 400MB)을 data/raw/에 받아 문서를 섞은 뒤 TARGET_CHARS 글자만큼 이어붙임.
- 글자 단위 토크나이저의 어휘가 너무 커지지 않도록, 드물게 나오는 글자(MIN_COUNT회 미만)는 삭제함.

사용법: python prepare_data.py
"""

import os
import random
import ssl
import urllib.request
from collections import Counter

import certifi
import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT/resolve/main/data/train-00000-of-00018.parquet"
RAW_PATH = "data/raw/webtext_00000.parquet"
OUT_PATH = "data/input.txt"
TARGET_CHARS = 30_000_000
MIN_COUNT = 100
MIN_DOC_CHARS = 300   # 너무 짧은 문서(메뉴, 광고 조각 등)는 제외

os.makedirs("data/raw", exist_ok=True)

if not os.path.exists(RAW_PATH):
    print("다운로드 중 (약 400MB)...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(URL, context=ctx) as r, open(RAW_PATH + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.rename(RAW_PATH + ".part", RAW_PATH)

articles = [a for a in pq.read_table(RAW_PATH, columns=["text"]).column("text").to_pylist()
            if len(a) >= MIN_DOC_CHARS]
random.Random(42).shuffle(articles)

picked, total = [], 0
for a in articles:
    picked.append(a.strip())
    total += len(a) + 2
    if total >= TARGET_CHARS:
        break
text = "\n\n".join(picked)[:TARGET_CHARS]

counts = Counter(text)
keep = {c for c, n in counts.items() if n >= MIN_COUNT}
text = "".join(c for c in text if c in keep)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(text)
print(f"문서 {len(picked):,}개 | 글자 수 {len(text):,} | 어휘 크기 {len(keep):,} → {OUT_PATH}")
