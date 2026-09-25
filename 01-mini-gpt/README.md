# 01 · 미니 GPT

GDGoC LLM 스터디에서 배운 토큰화, 임베딩, Self-Attention, 트랜스포머, 훈련 루프를 구현한 미니 GPT.

| 파일 | 설명 |
|---|---|
| `mini_gpt.ipynb` | 글자 단위 소형 GPT를 개념 단위로 나눠 설명을 붙인 노트북. 학습 데이터는 `data/input.txt` |
| `mini_gpt_mps.ipynb` | 같은 데이터로 모델을 키워 Apple GPU(MPS)에서 소형 모델과 성능·속도를 비교하는 노트북 |
| `mini_gpt_tuning.ipynb` | 학습률·모델 크기를 같은 시간 예산으로 튜닝하고 튜닝 전후를 비교하는 노트북 |
| `prepare_data.py` | 한국어 웹 텍스트(HAERAE-HUB/KOREAN-WEBTEXT)로 `data/input.txt`(약 3천만 자)를 만드는 스크립트. 위키백과는 각주·참고문헌이 많아 제외함 |
| `data/` | 생성된 데이터. 용량과 라이선스 문제로 git에 올리지 않음 |

## 실행
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter ipykernel
python prepare_data.py   # 데이터 생성 (원본 약 400MB 다운로드, 최초 1회)
```
VS Code에서 노트북을 열고 `.venv` 커널을 선택한 뒤 위에서부터 순서대로 실행함.
