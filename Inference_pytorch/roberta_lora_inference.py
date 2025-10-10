"""Run RoBERTa inference with an (optional) LoRA adapter under the NeuroSim flow.

This utility mirrors the existing computer-vision inference script but is tailored
for transformer-based NLP models from Hugging Face.  It demonstrates how to load a
RoBERTa backbone, merge a LoRA adapter with the base weights, and execute an
evaluation pass that records the layer-wise traces required by NeuroSim.

Example usage (GLUE/SST-2 with merged LoRA weights):

    python roberta_lora_inference.py \
        --dataset_name glue --dataset_config sst2 --split validation \
        --text_field sentence --label_field label \
        --base_model roberta-base \
        --lora_path /path/to/lora/adapter \
        --merge_lora 1 --max_samples 128 \
        --inference 1 --subArray 128 --parallelRead 128 --cellBit 1

The script expects the `datasets`, `transformers`, and `peft` packages to be
available in the current Python environment.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utee import hook, make_path, misc

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - informative error for users
    raise ImportError(
        "The `datasets` package is required for `roberta_lora_inference.py`. "
        "Install it with `pip install datasets`."
    ) from exc

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:  # pragma: no cover - informative error for users
    raise ImportError(
        "The `transformers` package is required for `roberta_lora_inference.py`. "
        "Install it with `pip install transformers`."
    ) from exc


def _load_peft_model(base_model, lora_path: Optional[str], merge_lora: bool):
    if not lora_path:
        return base_model

    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - informative error for users
        raise ImportError(
            "The `peft` package is required when providing `--lora_path`. "
            "Install it with `pip install peft`."
        ) from exc

    model = PeftModel.from_pretrained(base_model, lora_path)
    if merge_lora:
        # Merge the low-rank adapters into the base weights so that NeuroSim
        # captures the effective parameters.
        model = model.merge_and_unload()
    return model


@dataclass
class EncodedTextDataset(Dataset):
    encodings: Dict[str, torch.Tensor]
    labels: torch.Tensor

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RoBERTa(+LoRA) with NeuroSim")
    parser.add_argument("--base_model", default="roberta-base", help="Hugging Face model identifier")
    parser.add_argument(
        "--model_path",
        default=None,
        help="Optional local path to a pre-trained RoBERTa checkpoint (overrides --base_model)",
    )
    parser.add_argument("--lora_path", default=None, help="Path to a LoRA adapter checkpoint")
    parser.add_argument("--merge_lora", type=int, default=1, help="Merge LoRA weights into the base model (1|0)")
    parser.add_argument("--dataset_name", default="glue", help="Dataset hub name (e.g., glue)")
    parser.add_argument("--dataset_config", default="sst2", help="Dataset configuration (e.g., sst2)")
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate")
    parser.add_argument("--text_field", default="sentence", help="Field name for the primary text input")
    parser.add_argument("--text_pair_field", default=None, help="Optional second text field (sentence pairs)")
    parser.add_argument("--label_field", default="label", help="Field name containing labels")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit the number of evaluation samples")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=117, help="Random seed")
    parser.add_argument("--mode", default="FP", help="Computation mode to report to NeuroSim (FP or WAGE)")
    parser.add_argument("--model_tag", default="roberta_lora", help="Name used for NeuroSim logging artifacts")

    # Hardware parameters reused from the CV example.
    parser.add_argument("--inference", type=int, default=1, help="Run hardware inference simulation")
    parser.add_argument("--subArray", type=int, default=128, help="Size of sub-array (e.g. 128)")
    parser.add_argument("--parallelRead", type=int, default=128, help="Rows activated in parallel (<= subArray)")
    parser.add_argument("--ADCprecision", type=int, default=5, help="ADC precision (bits)")
    parser.add_argument("--cellBit", type=int, default=1, help="Cell precision (bits)")
    parser.add_argument("--wl_weight", type=int, default=8, help="Weight bit-width")
    parser.add_argument("--wl_activate", type=int, default=8, help="Activation bit-width")

    parser.add_argument("--vari", type=float, default=0.0, help="Conductance variation (std dev)")
    parser.add_argument("--t", type=float, default=0.0, help="Retention time")
    parser.add_argument("--v", type=float, default=0.0, help="Drift coefficient")
    parser.add_argument("--detect", type=int, default=0, help="Fixed-direction drift flag")
    parser.add_argument("--target", type=float, default=0.0, help="Drift target for fixed-direction drift")

    parser.add_argument("--logdir", default="log/roberta_lora", help="Directory to store logs")

    args = parser.parse_args()
    args.merge_lora = bool(args.merge_lora)
    args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
    args = make_path.makepath(args, ["logdir"])
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_model(args: argparse.Namespace, num_labels: int):
    if args.model_path:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=num_labels)

    model = _load_peft_model(model, args.lora_path, args.merge_lora)
    return model


def main() -> None:
    args = parse_args()
    misc.ensure_dir(args.logdir)
    misc.logger.init(args.logdir, "roberta_lora_log")
    logger = misc.logger.info

    logger("=================FLAGS==================")
    for k, v in args.__dict__.items():
        logger(f"{k}: {v}")
    logger("========================================")

    set_seed(args.seed)

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    split = args.split if args.split in dataset else next(iter(dataset.keys()))
    eval_dataset = dataset[split]
    if args.max_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_samples, len(eval_dataset))))

    label_column = args.label_field
    label_values = eval_dataset[label_column]
    if hasattr(eval_dataset.features[label_column], "names"):
        num_labels = len(eval_dataset.features[label_column].names)
    else:
        num_labels = len(set(label_values))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path or args.base_model)

    text_inputs = eval_dataset[args.text_field]
    if args.text_pair_field is not None:
        text_pairs = eval_dataset[args.text_pair_field]
    else:
        text_pairs = None

    encoded_inputs = tokenizer(
        text_inputs,
        text_pairs,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    labels = torch.tensor(label_values, dtype=torch.long)
    encoded_dataset = EncodedTextDataset(encoded_inputs, labels)

    dataloader = DataLoader(encoded_dataset, batch_size=args.batch_size)

    model = prepare_model(args, num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Attach NeuroSim hooks prior to the first forward pass.
    hook_handle_list = hook.hardware_evaluation(
        model,
        args.wl_weight,
        args.wl_activate,
        args.subArray,
        args.parallelRead,
        args.model_tag,
        args.mode,
    )

    total = 0
    correct = 0

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = logits.argmax(dim=-1)
            labels = batch["labels"]
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if step == 0:
                hook.remove_hook_list(hook_handle_list)

    accuracy = correct / total if total > 0 else 0.0
    logger(f"Evaluation accuracy: {accuracy * 100:.2f}%")

    if args.inference:
        logger(" --- Hardware Properties --- ")
        logger(f"subArray size: {args.subArray}")
        logger(f"parallel read: {args.parallelRead}")
        logger(f"ADC precision: {args.ADCprecision}")
        logger(f"cell precision: {args.cellBit}")
        logger(f"weight bit-width: {args.wl_weight}")
        logger(f"activation bit-width: {args.wl_activate}")
        logger(f"variation: {args.vari}")

        trace_cmd = os.path.join(
            os.getcwd(), f"layer_record_{args.model_tag}", "trace_command.sh"
        )
        if os.path.exists(trace_cmd):
            logger("Invoking NeuroSim backend via trace_command.sh")
            subprocess.run(["/bin/bash", trace_cmd], check=True)
        else:
            logger(f"Warning: Expected trace command not found at {trace_cmd}")


if __name__ == "__main__":
    main()

