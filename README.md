# Zero-shot steel defect recognition from small local LLM descriptions

Code for the paper "Zero-Shot Computer Vision for Steel Defects with Small Local Language Models: A Feasibility Study".

A local LLM (via Ollama) writes short appearance descriptions of each class. A text embedder turns them into class
anchors. A two-layer head on frozen MobileNetV3-Large features is trained on 47 DTD texture classes to map images onto
their anchors, then tested zero-shot on the six NEU steel defect classes (no NEU labels).

## Requirements
- Python 3.12, PyTorch, torchvision, transformers, numpy, pillow, matplotlib, psutil, datasets (for the DTD mirror)
- Ollama v0.40.8 with these models pulled: `nomic-embed-text`, `gemma3:1b`, `qwen3:1.7b`, `llama3.2:3b`, `gemma3:4b`, `qwen3:4b-instruct`
- CPU only is enough (all reported numbers were produced on a laptop CPU)

## Data (not included; please download)
- NEU surface defect database, NEU-CLS: 1800 grayscale images, 6 classes x 300 (original source: Song and Yan, 2013; He et al., 2020).
- DTD, 47 texture classes, 5640 images: `python dtd_from_hf.py --out path/to/dtd` downloads a Hugging Face mirror.

## Run order
```
# 1. descriptions (about 10 per defect class, 6 per texture class); sampling seeds are fixed
python step1_gen_descriptions.py --models gemma3:1b qwen3:1.7b llama3.2:3b gemma3:4b qwen3:4b-instruct --n 10
python t1_gen_texture_descriptions.py --dtd path/to/dtd --models gemma3:1b qwen3:1.7b llama3.2:3b gemma3:4b qwen3:4b-instruct --n 6
# 2. text embeddings
python step2_embed_text.py
python t2_embed_texture_text.py --dtd path/to/dtd
# 3. image features (frozen MobileNetV3-Large)
python step3_extract_features.py --data path/to/NEU
python t3_extract_dtd_features.py --dtd path/to/dtd
# 4. main experiment (tags = model name with ':' replaced by '_'; join tags with '+' to pool descriptions)
python t4_run_texture_zsl.py --tags llama3.2_3b qwen3_4b-instruct --seeds 5 --center
```
Each script has a usage note in its first lines. Defaults used in the paper: scale 20, lambda 1, hidden 512, dropout 0.3,
AdamW lr 1e-3, weight decay 1e-2, 100 full-batch epochs, 5 seeds (not tuned).

## Which script produces what
| Script | Paper content |
|---|---|
| `t4_run_texture_zsl.py` | Table I (text sources), ablations in Table II (`--lam 0`, `--hidden`, `--no-align`, `--max-desc`, no `--center`) |
| `t5_controls.py` | Shuffled-text controls (Table II) |
| `t6_supervised_reference.py` | Linear-probe reference with k labelled images per class (Table III) |
| `step5_bench_latency.py` | Latency and model size (Table III) |
| `step6_clip_baseline.py` | CLIP ViT-B/32 with a single prompt (Tables I and III) |
| `t10_clip_llm.py` | CLIP given the same LLM descriptions (Table I, last column) |
| `t9_leakage.py` | Removing related DTD classes: hand-picked, nearest-anchor and random (Table IV) |
| `t8_figures.py` | Confusion matrix figure |
| `anchor_diagnostics.py` | Anchor-similarity statistics mentioned in the Limitations |

## Earlier protocol (abandoned)
`step4_run_zsl.py`, `step1b_attributes.py`, `pairs_report.py`, `merge_embeds.py` belong to the first protocol (train on four NEU
classes, test on two). It was unstable and is mentioned only in the Limitations. `results.csv` is its output.

## Notes on the result files
Several scripts write a fixed output file name (`texture_results.csv`, `t9_leakage.csv`, `clip_llm_results.csv`), so a later run
overwrites an earlier one. The tables in the paper come from the console output of each command; the exact commands are above.
