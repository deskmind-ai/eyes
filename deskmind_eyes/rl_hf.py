"""GRPO grounding RL on one GPU with HF transformers + PEFT, mirroring rl.py on Tinker.

python -m deskmind_eyes.rl_hf --model models/Qwen3.5-4B --data showui_desktop \
    --data-dir data/showui_desktop --difficulty data/showui_desktop_lossless.jsonl --out runs/rl-hf
python -m deskmind_eyes.rl_hf ... --init-adapter runs/hf-sft-full-v3-lr1e-4/final   # RL on top of SFT
python -m deskmind_eyes.rl_hf --model models/GUI-Owl-1.5-4B-Instruct --prompt-style tool \
    --data showui_desktop,osatlas_full_v3 --data-dir ... --osatlas-dir ... --difficulty guiowl_difficulty.jsonl ...

Same algorithm and defaults as rl.py (round-2 recipe): per record sample `group` answers at temperature 1.0 (no
top-k / top-p), reward 1 if the point lands in the bbox, advantage = r - group mean, groups with identical rewards
are skipped, loss = -sum_t A * log p(token) per sample (Tinker's importance_sampling loss is exactly this when
sampling is on-policy), AdamW(0.9, 0.95, eps 1e-8, no decay), lr 4e-5 decayed linearly to 10% at max_steps,
32 records per step, images downscaled to ~1M pixels. Records, holdout and order match data.load_train_records.

Rollouts use HF `generate` on the policy itself, so there is no weight sync: slower than vLLM but exact.
The LoRA (r=32 on every language-model linear + lm_head) is train_hf's, so an SFT adapter can be continued.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image

from deskmind_eyes.common import MAX_PIXELS, grounding_messages, parse_point, resize_to_max_pixels
from deskmind_eyes.train_hf import build_model, env_report, latest_save, load_records, save


def in_bbox(point, bbox) -> bool:
    if point is None:
        return False
    x1, y1, x2, y2 = bbox
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def load_showui(data_dir: Path, seed: int) -> list[dict]:
    """data.load_train_records("showui_desktop", seed): 6-screenshot holdout, dedup by key, seeded shuffle."""
    screenshots = json.load(open(data_dir / "hf_train.json"))
    random.Random(0).shuffle(screenshots)
    recs = [{"source": "showui_desktop", "img_url": s["img_url"], "img_path": str(data_dir / "images" / s["img_url"]),
             "instruction": e["instruction"], "point": e["point"], "bbox": e["bbox"]}
            for s in screenshots[6:] for e in s["element"]]
    return finish(recs, seed)


def load_osatlas(records: Path, image_root: Path, seed: int) -> list[dict]:
    """data.load_train_records("osatlas_full_v3", seed)."""
    recs = load_records(str(records), str(image_root), holdout_images=30, seed=0)
    return finish(recs, seed)


def load_groundcua(records: Path, seed: int, holdout_images: int = 100) -> list[dict]:
    """prepare_groundcua records (img_path relative to the repo or absolute); a seeded screenshot-level holdout."""
    recs = [json.loads(l) for l in open(records)]
    images = sorted({r["img_filename"] for r in recs})
    random.Random(0).shuffle(images)
    held = set(images[:holdout_images])
    return finish([r for r in recs if r["img_filename"] not in held], seed)


def load_pools(names: str, args, seed: int) -> list[dict]:
    """Comma list of pools -> records (one shuffle over the union when several are given)."""
    recs = []
    for name in names.split(","):
        if name == "showui_desktop":
            recs += load_showui(Path(args.data_dir), seed)
        elif name == "osatlas_full_v3":
            d = Path(args.osatlas_dir or args.data_dir)
            recs += load_osatlas(d / "records_full_v3.jsonl", d, seed)
        elif name == "groundcua":
            recs += load_groundcua(Path(args.groundcua_records), seed)
        else:
            raise ValueError(name)
    if "," in names:
        random.Random(seed).shuffle(recs)
    return recs


def finish(recs: list[dict], seed: int) -> list[dict]:
    for r in recs:
        r["key"] = f"{r['source']}|{r.get('img_url') or r['img_filename']}|{r['instruction']}"
    recs = list({r["key"]: r for r in recs}.values())
    random.Random(seed).shuffle(recs)
    return recs


def filter_difficulty(recs: list[dict], path: str, lo: float, hi: float) -> list[dict]:
    rate = {}
    for line in open(path):
        d = json.loads(line)
        rate[d["key"]] = d["passes"] / d["samples"]
    kept = [r for r in recs if r["key"] in rate and lo <= rate[r["key"]] <= hi]
    print(f"difficulty filter [{lo}, {hi}]: {len(kept)} / {len(recs)} kept", flush=True)
    return kept


class Policy:
    def __init__(self, model, processor, max_pixels: int, prompt_style: str = "point"):
        self.model, self.proc, self.tok = model, processor, processor.tokenizer
        self.prompt_style = prompt_style
        self.tok.padding_side = "left"
        self.max_pixels = max_pixels
        self.eos = self.tok.convert_tokens_to_ids("<|im_end|>")

    def image(self, rec: dict) -> Image.Image:
        return resize_to_max_pixels(Image.open(rec["img_path"]).convert("RGB"), self.max_pixels)

    def prompt_text(self, rec: dict) -> str:
        messages = grounding_messages(rec["instruction"], self.prompt_style)
        return self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)

    @torch.no_grad()
    def rollout(self, recs: list[dict], group: int, temperature: float, max_tokens: int) -> list[list[list[int]]]:
        """Per record, `group` sampled completions (token ids up to and including <|im_end|>)."""
        self.model.eval()
        texts, images = [], []
        for r in recs:
            img, text = self.image(r), self.prompt_text(r)
            texts += [text] * group
            images += [img] * group
        inputs = self.proc(text=texts, images=images, padding=True, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                                  max_new_tokens=max_tokens, eos_token_id=self.eos,
                                  pad_token_id=self.tok.pad_token_id)
        gen = out[:, inputs["input_ids"].shape[1]:].tolist()
        seqs = []
        for g in gen:
            seqs.append(g[: g.index(self.eos) + 1] if self.eos in g else [t for t in g if t != self.tok.pad_token_id])
        return [seqs[i * group:(i + 1) * group] for i in range(len(recs))]

    def backward_group(self, rec: dict, seqs: list[list[int]], advs: list[float]) -> int:
        """Accumulate -A * sum log p(completion) for each sample of one record; returns trained tokens."""
        self.model.train()
        prompt = self.proc(text=[self.prompt_text(rec)], images=[self.image(rec)], return_tensors="pt").to("cuda")
        n = 0
        for seq, adv in zip(seqs, advs):
            if not seq or adv == 0:
                continue
            gen = torch.tensor([seq], device="cuda")
            ids = torch.cat([prompt["input_ids"], gen], dim=1)
            kw = {}
            for k, v in prompt.items():
                if k in ("input_ids", "attention_mask"):
                    continue
                if torch.is_tensor(v) and v.dim() == 2 and v.shape == prompt["input_ids"].shape:
                    v = torch.cat([v, torch.zeros_like(gen)], dim=1)  # per-token fields (mm_token_type_ids): text
                kw[k] = v
            logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=len(seq) + 1,
                                **kw).logits[:, :-1].float()
            logp = torch.log_softmax(logits, -1).gather(-1, gen.unsqueeze(-1)).squeeze(-1)
            (-adv * logp.sum()).backward()
            n += len(seq)
        return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="showui_desktop",
                    help="comma list of showui_desktop / osatlas_full_v3 / groundcua (pools are merged, then shuffled)")
    ap.add_argument("--data-dir", help="showui_desktop dir (or the osatlas_desktop dir when it is the only pool)")
    ap.add_argument("--osatlas-dir", help="osatlas_desktop dir holding records_full_v3.jsonl and images/")
    ap.add_argument("--groundcua-records", default="data/groundcua/records_functional.jsonl")
    ap.add_argument("--prompt-style", choices=["point", "tool", "owl"], default="point",
                    help="point: our point_2d prompt; tool: Qwen3-VL computer_use tool call (GUI-Owl's native format)")
    ap.add_argument("--difficulty", help="difficulty.py jsonl (base-model pass rates)")
    ap.add_argument("--min-pass", type=float, default=0.1)
    ap.add_argument("--max-pass", type=float, default=0.9)
    ap.add_argument("--init-adapter", help="continue this train_hf adapter dir (e.g. the SFT final)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--rollout-prompts", type=int, default=4, help="records per generate call")
    ap.add_argument("--dynamic-sampling", action="store_true",
                    help="DAPO: resample until batch-size groups have mixed rewards (fixes the late-run gradient drought)")
    ap.add_argument("--max-sample-factor", type=int, default=4, help="dynamic sampling: at most this x batch records/step")
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--lr-end-frac", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = load_pools(args.data, args, args.seed)
    if args.difficulty:
        recs = filter_difficulty(recs, args.difficulty, args.min_pass, args.max_pass)

    model, processor = build_model(args.model, grad_ckpt=True)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        res = set_peft_model_state_dict(model, load_file(Path(args.init_adapter) / "adapter_model.safetensors"))
        assert not getattr(res, "unexpected_keys", None), res
        print(f"initialized LoRA from {args.init_adapter}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    policy = Policy(model, processor, args.max_pixels, args.prompt_style)
    (out / "env.json").write_text(json.dumps({**env_report(), **vars(args), "records": len(recs)}, indent=2))
    print(json.dumps({"records": len(recs), **env_report()}), flush=True)

    start = 0
    if (last := latest_save(out)) is not None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(last / "adapter_model.safetensors"))
        opt.load_state_dict(torch.load(last / "optimizer.pt"))
        start = json.loads((last / "trainer_state.json").read_text())["step"]
        print(f"resumed from {last} at step {start}", flush=True)

    # Record cursor: step * batch_size without dynamic sampling; otherwise the records consumed so far (from the log).
    cursor = start * args.batch_size
    if args.dynamic_sampling and (out / "metrics.jsonl").exists():
        cursor = sum(json.loads(l).get("records_seen", args.batch_size) for l in open(out / "metrics.jsonl")
                     if json.loads(l)["step"] < start)
    log = open(out / "metrics.jsonl", "a")
    for step in range(start, args.max_steps):
        t0 = time.time()
        lr = args.lr * (1 - (1 - args.lr_end_frac) * step / max(1, args.max_steps - 1))
        for g in opt.param_groups:
            g["lr"] = lr
        # Rollouts: a fixed slice of records per step, or with --dynamic-sampling (DAPO) keep drawing records until
        # batch_size groups have mixed rewards (uniform groups carry no gradient), up to max_sample_factor x batch.
        budget = args.batch_size * (args.max_sample_factor if args.dynamic_sampling else 1)
        rewards, kept, n_groups, n_right, n_wrong, n_fail = [], [], 0, 0, 0, 0
        while n_groups < budget and (not args.dynamic_sampling or len(kept) < args.batch_size):
            n = min(args.rollout_prompts, budget - n_groups)
            chunk = [recs[(cursor + i) % len(recs)] for i in range(n)]
            cursor += n
            for rec, seqs in zip(chunk, policy.rollout(chunk, args.group, args.temperature, args.max_tokens)):
                n_groups += 1
                rs = []
                for sq in seqs:
                    point = parse_point(policy.tok.decode(sq, skip_special_tokens=True))
                    n_fail += point is None
                    rs.append(float(in_bbox(point, rec["bbox"])))
                rewards += rs
                mean = sum(rs) / len(rs)
                if mean == 1.0:
                    n_right += 1
                elif mean == 0.0:
                    n_wrong += 1
                else:
                    kept.append((rec, seqs, [r - mean for r in rs]))
        kept = kept[: args.batch_size]
        t_sample = time.time() - t0

        t1 = time.time()
        tokens = 0
        if kept:
            for rec, seqs, advs in kept:
                tokens += policy.backward_group(rec, seqs, advs)
            opt.step()
            opt.zero_grad(set_to_none=True)
        m = {"step": step, "lr": lr, "reward/mean": sum(rewards) / len(rewards), "records_seen": n_groups,
             "groups/all_right_frac": n_right / n_groups, "groups/all_wrong_frac": n_wrong / n_groups,
             "groups/trained": len(kept), "groups/trained_frac": len(kept) / n_groups,
             "sample/parse_fail_frac": n_fail / len(rewards),
             "train/gen_tokens": tokens, "time/sample": t_sample, "time/train": time.time() - t1,
             "time/step": time.time() - t0, "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        log.write(json.dumps(m) + "\n")
        log.flush()
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
        if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < args.max_steps:
            save(model, opt, out, step + 1, f"step_{step + 1:06d}")
    save(model, opt, out, args.max_steps, "final")
    print(f"done: {args.max_steps} steps -> {out / 'final'}", flush=True)


if __name__ == "__main__":
    main()
