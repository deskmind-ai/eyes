"""LoRA SFT of Qwen3.5-4B with HF transformers + PEFT on a single GPU, mirroring train.py.

python -m deskmind_eyes.train_hf --model models/Qwen3.5-4B --records data/osatlas_desktop/records.jsonl \
    --image-root data/osatlas_desktop --out runs/hf-sft-osatlas
python -m deskmind_eyes.train_hf --model models/Qwen3.5-4B --synthetic 16 --max-steps 3 --batch-size 4 \
    --out /tmp/probe      # probe: checks the linear-attention fast path and times fwd/bwd, no data needed

Matches the Tinker run (train.py + cookbook defaults): LoRA r=32 / alpha=32 / no dropout on every
language-model linear layer plus lm_head (vision tower frozen), AdamW(0.9, 0.95, eps 1e-8, no weight
decay), lr 4.9e-4 decayed linearly to 0 over the run, 32 samples per step, images downscaled to
~1M pixels, the eval prompt, and loss only on the answer tokens (normalized over the whole batch).
One difference is structural: HF fuses the linear-attention q/k/v projection (`in_proj_qkv`) where
Tinker adapts q, k and v separately.

Adapter + optimizer state are saved every --save-every steps and the run resumes from the latest save.
"""

import argparse
import json
import math
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image

from deskmind_eyes.common import MAX_PIXELS, NATIVE_MAX_PIXELS, PROMPT, format_answer, resize_to_max_pixels

LORA_TARGETS = (r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
                r"|in_proj_qkv|in_proj_z|out_proj)$|^lm_head$")
DEFAULT_LR = 4.905e-4  # tinker_cookbook.hyperparam_utils.get_lr("Qwen/Qwen3.5-4B")


def load_records(path: str, image_root: str, holdout_images: int, seed: int) -> list[dict]:
    """Same screenshot-level holdout as data.load_osatlas, with image paths rebased onto image_root."""
    records = [json.loads(l) for l in open(path)]
    images = sorted({r["img_path"] for r in records})
    random.Random(seed).shuffle(images)
    held = set(images[:holdout_images])
    out = []
    for r in records:
        if r["img_path"] in held:
            continue
        rel = r["img_path"].split("osatlas_desktop/", 1)[-1]  # records store absolute paths from the Mac
        out.append({**r, "img_path": str(Path(image_root) / rel)})
    return out


def synthetic_records(n: int) -> list[dict]:
    rng = random.Random(0)
    return [{"img_path": None, "size": (1920, 1080), "instruction": f"Settings button {i}",
             "point": [rng.random(), rng.random()]} for i in range(n)]


IMAGE_MAX_PIXELS = MAX_PIXELS  # set from --max-pixels; NATIVE_MAX_PIXELS keeps screenshots as the eval sees them


def load_image(rec: dict) -> Image.Image:
    if rec["img_path"] is None:  # synthetic: noise screenshot of the requested size
        return resize_to_max_pixels(Image.effect_noise(rec["size"], 64).convert("RGB"), IMAGE_MAX_PIXELS)
    return resize_to_max_pixels(Image.open(rec["img_path"]).convert("RGB"), IMAGE_MAX_PIXELS)


class Encoder:
    """Prompt exactly as eval_vllm / the HF chat template renders it (thinking off), answer appended."""

    def __init__(self, processor):
        self.processor = processor
        self.tok = processor.tokenizer
        self.tok.padding_side = "left"
        self.lock = threading.Lock()  # HF fast tokenizers raise "Already borrowed" under concurrent calls

    def answer_ids(self, rec: dict) -> list[int]:
        with self.lock:
            return self.tok(format_answer(*rec["point"]) + "<|im_end|>", add_special_tokens=False).input_ids

    def __call__(self, recs: list[dict]) -> dict:
        """Left-padded micro-batch, so every answer ends at the last position and the loss only needs
        logits for the final max(n_ans) positions (the LM head over the whole ~1K-token image prompt
        is ~13% of the compute and a 1GB fp32 tensor otherwise)."""
        texts, answers = [], []
        for rec in recs:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": PROMPT.format(instruction=rec["instruction"])}]}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                        enable_thinking=False)
            answers.append(format_answer(*rec["point"]) + "<|im_end|>")
            texts.append(prompt + answers[-1])
        images = [load_image(r) for r in recs]  # PNG decode + resize: the slow part, outside the lock
        with self.lock:
            batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = torch.full_like(batch["input_ids"], -100)
        for i, (rec, answer) in enumerate(zip(recs, answers)):
            n = len(self.answer_ids(rec))
            labels[i, -n:] = batch["input_ids"][i, -n:]
            # The answer must tokenize the same inside the sequence as on its own, or the mask is off.
            with self.lock:
                assert self.tok.decode(batch["input_ids"][i, -n:]) == answer, "answer/prompt token boundary moved"
        batch["labels"] = labels
        return batch


def build_model(model_path: str, grad_ckpt: bool = True):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa").cuda()
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0,
                                             target_modules=LORA_TARGETS, bias="none"))
    return model, processor


def env_report() -> dict:
    import importlib.util

    try:  # Qwen3.5's linear-attention kernels; other bases (e.g. Qwen3-VL / GUI-Owl) have none
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    except ImportError:
        m = None
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "fla": importlib.util.find_spec("fla") is not None,
        "causal_conv1d": importlib.util.find_spec("causal_conv1d") is not None,
        "linear_attn_fast_path": bool(getattr(m, "is_fast_path_available", False)),
    }


def latest_save(out: Path) -> Path | None:
    saves = sorted(p for p in out.glob("step_*") if (p / "trainer_state.json").exists())
    return saves[-1] if saves else None


def save(model, opt, out: Path, step: int, name: str) -> None:
    d = out / name
    model.save_pretrained(d, save_embedding_layers=False)  # LoRA on lm_head only; the tied embedding stays frozen
    torch.save(opt.state_dict(), d / "optimizer.pt")
    (d / "trainer_state.json").write_text(json.dumps({"step": step}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local HF dir of Qwen3.5-4B")
    ap.add_argument("--records", help="records.jsonl from prepare_osatlas.py")
    ap.add_argument("--image-root", help="directory holding images/<os>/... (the osatlas_desktop dir)")
    ap.add_argument("--synthetic", type=int, default=0, help="use N synthetic samples instead of --records")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--holdout-images", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro-batch", type=int, default=1, help="samples per forward pass (left-padded)")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="skip activation recompute (more memory)")
    ap.add_argument("--limit", type=int, help="only use the first N records (benchmarking)")
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS,
                    help=f"downscale images above this (default {MAX_PIXELS} as on Tinker; "
                         f"{NATIVE_MAX_PIXELS} = native resolution, matching the Pro eval)")
    args = ap.parse_args()
    global IMAGE_MAX_PIXELS
    IMAGE_MAX_PIXELS = args.max_pixels

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = (synthetic_records(args.synthetic) if args.synthetic
               else load_records(args.records, args.image_root, args.holdout_images, args.seed))
    records = records[: args.limit] if args.limit else records
    steps_per_epoch = len(records) // args.batch_size
    total = min(args.max_steps or 10**9, steps_per_epoch * args.epochs)
    order = []
    for e in range(args.epochs):
        idx = list(range(len(records)))
        random.Random(args.seed + e).shuffle(idx)
        order += idx[: steps_per_epoch * args.batch_size]

    model, processor = build_model(args.model, grad_ckpt=not args.no_grad_ckpt)
    encode = Encoder(processor)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    report = {**env_report(), "records": len(records), "steps": total, "micro_batch": args.micro_batch,
              "max_pixels": args.max_pixels, "lr": args.lr,
              "grad_ckpt": not args.no_grad_ckpt,
              "trainable_params": sum(p.numel() for p in params)}
    print(json.dumps(report), flush=True)
    (out / "env.json").write_text(json.dumps(report, indent=2))

    start = 0
    if (last := latest_save(out)) is not None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(last / "adapter_model.safetensors"))
        opt.load_state_dict(torch.load(last / "optimizer.pt"))
        start = json.loads((last / "trainer_state.json").read_text())["step"]
        print(f"resumed from {last} at step {start}", flush=True)

    model.train()
    log = open(out / "metrics.jsonl", "a")
    # Image decoding + processor run on CPU threads a few micro-batches ahead of the GPU.
    micro = [(step, order[step * args.batch_size + j : step * args.batch_size + j + args.micro_batch])
             for step in range(start, total) for j in range(0, args.batch_size, args.micro_batch)]
    pool = ThreadPoolExecutor(4)
    pending = deque(pool.submit(encode, [records[i] for i in idx]) for _, idx in micro[:4])
    next_submit = len(pending)
    for step in range(start, total):
        t0 = time.time()
        lr = args.lr * (1 - step / total)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = [records[i] for i in order[step * args.batch_size : (step + 1) * args.batch_size]]
        n_ans = sum(len(encode.answer_ids(r)) for r in batch)  # normalize over the whole batch
        loss_sum, tokens, padded, wait = 0.0, 0, 0, 0.0
        for _ in range(0, args.batch_size, args.micro_batch):
            tw = time.time()
            enc = pending.popleft().result()
            wait += time.time() - tw
            if next_submit < len(micro):
                pending.append(pool.submit(encode, [records[i] for i in micro[next_submit][1]]))
                next_submit += 1
            inputs = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            labels = inputs.pop("labels")
            keep = int((labels != -100).sum(1).max()) + 1  # answers end at the last position (left pad)
            logits = model(**inputs, logits_to_keep=keep).logits[:, :-1].float()
            nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                                    labels[:, -(keep - 1):].reshape(-1), ignore_index=-100,
                                                    reduction="sum")
            (nll / n_ans).backward()
            loss_sum += nll.item()
            tokens += int(inputs["attention_mask"].sum())
            padded += inputs["input_ids"].numel()
        opt.step()
        opt.zero_grad(set_to_none=True)
        dt = time.time() - t0
        m = {"step": step + 1, "lr": lr, "train_mean_nll": loss_sum / n_ans, "num_tokens": tokens,
             "pad_frac": 1 - tokens / padded, "time/step": dt, "time/data_wait": wait,
             "tokens_per_s": tokens / dt, "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        log.write(json.dumps(m) + "\n")
        log.flush()
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
        if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < total:
            save(model, opt, out, step + 1, f"step_{step + 1:06d}")
    save(model, opt, out, total, "final")
    print(f"done: {total} steps -> {out / 'final'}", flush=True)


if __name__ == "__main__":
    main()
