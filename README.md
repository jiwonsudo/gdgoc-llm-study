<div align="center">

# 🧠 GDGoC LLM Study

**GDGoC on Sangmyung AI 팀 LLM 스터디 — 배운 내용을 직접 구현해 보는 저장소**

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter">
<img src="https://img.shields.io/badge/Apple%20MPS-000000?style=flat-square&logo=apple&logoColor=white" alt="Apple MPS">
<img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=white" alt="Hugging Face">

</div>

## 실습 목록

| # | 주제 | 내용 |
|:-:|---|---|
| 01 | [미니 GPT](01-mini-gpt/) | 토큰화 → 임베딩 → Self-Attention → 트랜스포머 → 훈련 루프를 직접 구현한 글자 단위 소형 GPT. 한국어 웹 텍스트(약 3천만 자)로 학습하고, Apple GPU(MPS)에서 모델 크기별 성능·속도 비교와 학습률 튜닝까지 진행 |

## 빠르게 시작하기

```bash
cd 01-mini-gpt
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter ipykernel
python prepare_data.py          # 학습 데이터 생성 (최초 1회)
```

노트북을 순서대로 실행한 뒤, 저장된 모델로 글을 이어 쓸 수 있습니다.

```bash
python generate.py "오늘은 "
```

각 실습의 자세한 설명은 폴더 안의 README에 있습니다.
