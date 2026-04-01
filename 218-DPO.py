import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────
# 1. 数据集
# ─────────────────────────────────────────
@dataclass
class DPOSample:
    prompt: str
    chosen: str
    rejected: str


class DPODataset(Dataset):
    def __init__(self, samples: list[DPOSample], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def tokenize(self, prompt: str, response: str):
        """把 prompt+response 拼接后 tokenize，返回 input_ids 和 labels。
        labels 中 prompt 部分设为 -100（不参与 loss 计算）。
        """
        full_text = prompt + response
        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = enc["input_ids"].clone()
        labels[0, :prompt_len] = -100          # 屏蔽 prompt 部分
        labels[labels == self.tokenizer.pad_token_id] = -100  # 屏蔽 padding

        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), labels.squeeze(0)

    def __getitem__(self, idx):
        s = self.samples[idx]
        c_ids, c_mask, c_labels = self.tokenize(s.prompt, s.chosen)
        r_ids, r_mask, r_labels = self.tokenize(s.prompt, s.rejected)
        return {
            "chosen_input_ids":      c_ids,
            "chosen_attention_mask": c_mask,
            "chosen_labels":         c_labels,
            "rejected_input_ids":    r_ids,
            "rejected_attention_mask": r_mask,
            "rejected_labels":       r_labels,
        }


# ─────────────────────────────────────────
# 2. 核心工具函数
# ─────────────────────────────────────────
def get_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算每个序列的 token 级对数概率之和（只对非 -100 的位置求和）。
    logits : (B, L, V)
    labels : (B, L)   -100 表示忽略
    return : (B,)
    """
    log_probs = F.log_softmax(logits, dim=-1)          # (B, L, V)
    # shift：logits[t] 预测 label[t+1]
    shift_log_probs = log_probs[:, :-1, :]             # (B, L-1, V)
    shift_labels    = labels[:, 1:]                    # (B, L-1)

    mask = (shift_labels != -100).float()
    shift_labels = shift_labels.clamp(min=0)           # 避免 gather 时 -100 报错

    token_log_probs = shift_log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)                                      # (B, L-1)

    return (token_log_probs * mask).sum(dim=-1)        # (B,)


def dpo_loss(
        policy_model,
        ref_model,
        batch: dict,
        beta: float = 0.1,
        device: str = "cuda",
) -> tuple[torch.Tensor, dict]:
    """
    计算一个 batch 的 DPO loss。
    返回 (loss, metrics_dict)。
    """
    def forward(model, input_ids, attention_mask, labels):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return get_log_probs(out.logits, labels)

    # ── policy log probs ──
    pi_log_chosen   = forward(policy_model,
                              batch["chosen_input_ids"].to(device),
                              batch["chosen_attention_mask"].to(device),
                              batch["chosen_labels"].to(device))
    pi_log_rejected = forward(policy_model,
                              batch["rejected_input_ids"].to(device),
                              batch["rejected_attention_mask"].to(device),
                              batch["rejected_labels"].to(device))

    # ── reference log probs（不需要梯度）──
    with torch.no_grad():
        ref_log_chosen   = forward(ref_model,
                                   batch["chosen_input_ids"].to(device),
                                   batch["chosen_attention_mask"].to(device),
                                   batch["chosen_labels"].to(device))
        ref_log_rejected = forward(ref_model,
                                   batch["rejected_input_ids"].to(device),
                                   batch["rejected_attention_mask"].to(device),
                                   batch["rejected_labels"].to(device))

    # ── DPO loss ──
    # log ratio: log(π/π_ref)
    log_ratio_chosen   = pi_log_chosen   - ref_log_chosen     # (B,)
    log_ratio_rejected = pi_log_rejected - ref_log_rejected   # (B,)

    # Bradley-Terry margin
    margin = beta * (log_ratio_chosen - log_ratio_rejected)   # (B,)
    loss   = -F.logsigmoid(margin).mean()

    metrics = {
        "loss":           loss.item(),
        "reward_chosen":  log_ratio_chosen.mean().item(),
        "reward_rejected": log_ratio_rejected.mean().item(),
        "reward_margin":  (log_ratio_chosen - log_ratio_rejected).mean().item(),
        "accuracy":       (margin > 0).float().mean().item(),
    }
    return loss, metrics


# ─────────────────────────────────────────
# 3. 训练循环
# ─────────────────────────────────────────
def train_dpo(
        model_name: str = "gpt2",
        beta: float = 0.1,
        lr: float = 1e-5,
        epochs: int = 3,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # policy model（可训练）
    policy = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    # reference model（冻结，保持 SFT 初始化）
    ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── 示例数据 ──
    raw_data = [
        DPOSample(
            prompt="Human: 什么是机器学习？\nAssistant:",
            chosen=" 机器学习是让计算机从数据中自动学习规律的技术，无需显式编程。",
            rejected=" 机器学习就是让机器变聪明。",
        ),
        DPOSample(
            prompt="Human: 如何写好一份简历？\nAssistant:",
            chosen=" 简历应突出量化成果，根据岗位定制，保持简洁一到两页。",
            rejected=" 写简历就把你的经历写上去就好了。",
        ),
    ]

    dataset    = DPODataset(raw_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = torch.optim.AdamW(policy.parameters(), lr=lr)

    policy.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss, metrics = dpo_loss(policy, ref_model, batch, beta=beta, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            print(
                f"Epoch {epoch+1} Step {step+1} | "
                f"loss={metrics['loss']:.4f} | "
                f"margin={metrics['reward_margin']:.4f} | "
                f"acc={metrics['accuracy']:.2%}"
            )

    return policy, tokenizer


if __name__ == "__main__":
    policy_model, tok = train_dpo()