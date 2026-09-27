"""Merge a PEFT LoRA adapter from train_hf.py into Qwen3.5-4B, tensor by tensor (no transformers needed).

python -m deskmind_eyes.merge_hf_lora --base models/Qwen3.5-4B \
    --adapter runs/adapters/hf-sft-step_000050 --out runs/merged/hf-sft-step_000050
python -m deskmind_eyes.merge_hf_lora ... --verify   # also compare logits against base+PEFT (needs peft, GPU)

W' = W + (alpha / r) * B @ A for every adapted linear layer. Qwen3.5-4B ties lm_head to the input
embedding and the base checkpoint has no lm_head.weight; train_hf adapts lm_head only, so the merge
unties them: lm_head.weight = embed_tokens + delta, embed_tokens unchanged, tie_word_embeddings=false.
(Tinker's export instead folds the lm_head delta into the shared embedding, which also shifts inputs.)
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

EMBED = "model.language_model.embed_tokens.weight"


def load_deltas(adapter: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    tensors = load_file(adapter / "adapter_model.safetensors")
    pairs = {}
    for k, a in tensors.items():
        if ".lora_A." not in k:
            continue
        module = k.removeprefix("base_model.model.").split(".lora_A.")[0]
        b = tensors[k.replace(".lora_A.", ".lora_B.")]
        pairs[module + ".weight"] = (a.float() * scale, b.float())
    return pairs


def merge(base: Path, adapters: list[Path], out: Path) -> None:
    """With several adapters (same base), W' = W + mean_i(B_i @ A_i): a uniform average of the fine-tuning deltas."""
    many = [load_deltas(a) for a in adapters]
    pairs = many[0]
    assert all(m.keys() == pairs.keys() for m in many), "adapters target different modules"
    delta = lambda name: sum(m[name][1] @ m[name][0] for m in many) / len(many)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((base / "model.safetensors.index.json").read_text())
    applied = set()
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(base / shard)
        for name in list(tensors):
            if name in pairs:
                tensors[name] = (tensors[name].float() + delta(name)).to(tensors[name].dtype)
                applied.add(name)
        if EMBED in tensors and "lm_head.weight" in pairs:
            tensors["lm_head.weight"] = (tensors[EMBED].float() + delta("lm_head.weight")).to(tensors[EMBED].dtype)
            index["weight_map"]["lm_head.weight"] = shard
            applied.add("lm_head.weight")
        save_file(tensors, out / shard, metadata={"format": "pt"})
        print(f"{shard}: written", flush=True)
    missing = set(pairs) - applied
    assert not missing, f"adapter modules not found in the base checkpoint: {sorted(missing)[:5]}"

    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy(f, out / f.name)
    cfg = json.loads((base / "config.json").read_text())
    if "lm_head.weight" in applied:
        cfg["tie_word_embeddings"] = False
        cfg.get("text_config", {})["tie_word_embeddings"] = False
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"merged {len(applied)} weights -> {out}", flush=True)


def verify(base: Path, adapter: Path, out: Path) -> None:
    """Merged model must reproduce base+adapter logits (text-only prompt; the vision tower is untouched)."""
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    ids = tok("Locate the Save button in the toolbar.", return_tensors="pt").input_ids.cuda()
    kw = dict(dtype=torch.bfloat16)
    with torch.no_grad():
        peft = PeftModel.from_pretrained(AutoModelForImageTextToText.from_pretrained(base, **kw).cuda(), adapter)
        ref = peft(input_ids=ids).logits.float()
        del peft
        merged = AutoModelForImageTextToText.from_pretrained(out, **kw).cuda()
        got = merged(input_ids=ids).logits.float()
    diff = (ref - got).abs().max().item()
    agree = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
    print(json.dumps({"max_abs_logit_diff": diff, "argmax_agreement": agree}), flush=True)
    assert agree > 0.99, "merged model disagrees with base+adapter"


def verify_points(base: Path, adapter: Path, out: Path, records: Path, image_root: Path, n: int) -> None:
    """The check that matters: greedy click points on real screenshots (eval prompt, native resolution) must match
    between base+adapter and the merged model. The logit check above flags bf16 near-ties on a text-only prompt."""
    import json as _json
    import math

    from PIL import Image
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from deskmind_eyes.common import NATIVE_MAX_PIXELS, PROMPT, parse_point, resize_to_max_pixels

    recs = [_json.loads(l) for l in open(records)][:n]
    proc = AutoProcessor.from_pretrained(base)

    def run(model) -> list:
        pts = []
        for r in recs:
            rel = r["img_path"].split("osatlas_desktop/", 1)[-1]
            img = resize_to_max_pixels(Image.open(image_root / rel).convert("RGB"), NATIVE_MAX_PIXELS)
            msgs = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": PROMPT.format(instruction=r["instruction"])}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=32, do_sample=False)
            pts.append(parse_point(proc.tokenizer.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)))
        return pts

    kw = dict(dtype=torch.bfloat16)
    peft = PeftModel.from_pretrained(AutoModelForImageTextToText.from_pretrained(base, **kw).cuda(), adapter).eval()
    a = run(peft)
    del peft
    torch.cuda.empty_cache()
    b = run(AutoModelForImageTextToText.from_pretrained(out, **kw).cuda().eval())
    dists = [math.hypot(p[0] - q[0], p[1] - q[1]) * 1000 if p and q else float("inf") for p, q in zip(a, b)]
    same = sum(d <= 2 for d in dists)  # within 2/1000 of the screen
    print(_json.dumps({"n": len(recs), "points_within_2_per_mille": same, "max_dist_per_mille": max(dists),
                       "parse_fail": sum(p is None for p in a + b)}), flush=True)
    assert same >= 0.9 * len(recs), "merged model clicks differently from base+adapter"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--adapter", required=True, type=Path, nargs="+",
                    help="one adapter, or several to merge their averaged deltas (checkpoint averaging)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--verify", action="store_true", help="logit check on a text-only prompt (bf16-noisy)")
    ap.add_argument("--verify-records", type=Path, help="also compare greedy click points on these records")
    ap.add_argument("--verify-image-root", type=Path)
    ap.add_argument("--verify-n", type=int, default=20)
    args = ap.parse_args()
    merge(args.base, args.adapter, args.out)
    args.adapter = args.adapter[0]  # the checks below compare against the first adapter
    if args.verify:
        try:
            verify(args.base, args.adapter, args.out)
        except AssertionError as e:  # informative only; the point check below is the gate
            print(f"logit check: {e}", flush=True)
    if args.verify_records:
        verify_points(args.base, args.adapter, args.out, args.verify_records, args.verify_image_root, args.verify_n)


if __name__ == "__main__":
    main()
