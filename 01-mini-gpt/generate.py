"""
학습한 미니 GPT를 불러와서, 입력한 앞글자 뒤로 이어지는 글자를 생성하는 프로그램.

사전 준비: mini_gpt_mps.ipynb를 끝까지 실행해 models/ 에 가중치를 저장해 둘 것.

사용법:
    python generate.py                       # 대화형: 앞글자를 입력할 때마다 이어서 생성
    python generate.py "오늘은 "             # 한 번만 생성하고 종료
    python generate.py --model models/mini_gpt_small.pt --length 300 --temperature 0.7
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 모델 정의: mini_gpt_mps.ipynb와 동일해야 저장한 가중치를 불러올 수 있음 ──
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, n_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_length, n_heads, n_layers):
        super().__init__()
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_length, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.token_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        h = self.ln_f(self.blocks(h))
        return self.head(h)


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    chars = ckpt["chars"]
    cfg = ckpt["config"]
    model = MiniGPT(len(chars), cfg["embed_dim"], cfg["context_length"], cfg["n_heads"], cfg["n_layers"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), chars, ckpt.get("val_loss")


@torch.no_grad()
def generate(model, stoi, itos, prompt, length, temperature, top_k, device):
    ids = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    for _ in range(length):
        logits = model(ids[:, -model.context_length:])[:, -1, :] / temperature
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return "".join(itos[i] for i in ids[0].tolist())


def main():
    ap = argparse.ArgumentParser(description="앞글자를 입력하면 뒷글자를 생성함")
    ap.add_argument("prompt", nargs="?", help="앞글자. 생략하면 대화형으로 실행")
    ap.add_argument("--model", default="models/mini_gpt_scaleup.pt", help="저장한 모델 파일")
    ap.add_argument("--length", type=int, default=200, help="새로 생성할 글자 수")
    ap.add_argument("--temperature", type=float, default=0.8, help="낮을수록 무난하게, 높을수록 다양하게 생성")
    ap.add_argument("--top-k", type=int, default=40, help="확률 상위 k개 글자 중에서만 선택 (0이면 제한 없음)")
    ap.add_argument("--seed", type=int, default=None, help="지정하면 같은 입력에 같은 결과가 나옴")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, chars, val_loss = load_model(args.model, device)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    if args.seed is not None:
        torch.manual_seed(args.seed)

    def run(prompt):
        unknown = sorted({c for c in prompt if c not in stoi})
        if unknown:
            print(f"어휘에 없는 글자는 제외함: {' '.join(repr(c) for c in unknown)}")
            prompt = "".join(c for c in prompt if c in stoi)
        if not prompt:
            print("입력할 수 있는 글자가 없음")
            return
        print(generate(model, stoi, itos, prompt, args.length, args.temperature, args.top_k, device))

    if args.prompt is not None:
        run(args.prompt)
        return

    val = f", val loss {val_loss:.3f}" if val_loss is not None else ""
    print(f"모델: {args.model} (어휘 {len(chars)}자{val}, 장치 {device})")
    print("앞글자를 입력하면 이어서 생성함. 종료: 빈 줄 입력 또는 Ctrl-D")
    while True:
        try:
            prompt = input("\n앞글자> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            break
        run(prompt)


if __name__ == "__main__":
    main()
