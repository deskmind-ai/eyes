"""Download a Tinker LoRA checkpoint and merge it into a full HF model (for vLLM / official eval).

python -m deskmind_eyes.export_ckpt tinker://<run>:train:0/sampler_weights/final --name p1-filtered-final
-> runs/merged/<name>/  (config, tokenizer, processor, safetensors)

The LoRA from Tinker only touches the language model (incl. Qwen3.5 linear-attention layers), which
vLLM's LoRA path may not support, so we merge instead of serving the adapter. Needs TINKER_API_KEY
and the base model locally (e.g. `hf download Qwen/Qwen3.5-4B`) or on the HF hub.
"""

import argparse
import os
import shutil
from pathlib import Path

from tinker_cookbook import weights

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tinker_path")
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-model", default=os.environ.get("MODEL_DIR", str(Path.home() / "models" / "Qwen3.5-4B")),
                    help="local HF dir of Qwen/Qwen3.5-4B (or the hub id)")
    ap.add_argument("--keep-adapter", action="store_true")
    args = ap.parse_args()

    adapter_dir = ROOT / "runs" / "adapters" / args.name
    merged_dir = ROOT / "runs" / "merged" / args.name
    if merged_dir.exists():
        print(f"already merged: {merged_dir}")
        return
    weights.download(tinker_path=args.tinker_path, output_dir=str(adapter_dir))
    weights.build_hf_model(base_model=args.base_model, adapter_path=str(adapter_dir), output_path=str(merged_dir))
    if not args.keep_adapter:
        shutil.rmtree(adapter_dir, ignore_errors=True)
    print(f"merged model: {merged_dir}")


if __name__ == "__main__":
    main()
