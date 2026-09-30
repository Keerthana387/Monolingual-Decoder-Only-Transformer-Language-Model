# Nepali Monolingual Transformer Language Model

This directory contains the Nepali implementation for the LMA Individual Project. It includes the Nepali tokenizer pipeline, tokenized-corpus preparation, decoder-only Transformer implementation, pretraining pipeline, Phase 2 evaluation, synthetic reasoning-data generation, Phase 3 reasoning fine-tuning, and post-fine-tuning evaluation.

> **Scope:** This README documents the code, file organization, dependencies, inputs/outputs, and reproduction procedure. Results and interpretation are kept in the report.

---

## 1. Directory Structure

```text
Nepali/
├── config/
│   ├── __init__.py
│   ├── model_config.py
│   └── train_config.py
│
├── data/
│   ├── dataset.py
│   ├── dataset_validation.py
│   ├── prepare_validation.py
│   ├── tokenize_corpus.py
│   └── validation_dataset.py
│
├── model/
│   ├── __init__.py
│   ├── attention.py
│   ├── lm.py
│   ├── positional_embedding.py
│   ├── transformer.py
│   └── transformer_block.py
│
├── eval/
│   ├── attention.py
│   ├── common.py
│   ├── evaluate_lm.py
│   ├── generate.py
│   ├── metrics.py
│   ├── make_generation_samples.py
│   ├── plot_attention_local_global.py
│   ├── plot_attention_summary.py
│   ├── plot_generation_metrics.py
│   ├── plot_loss.py
│   ├── plot_lr_schedule.py
│   ├── plot_perplexity.py
│   └── finetune/
│       ├── attention_compare.py
│       ├── attention_compare_mixed.py
│       ├── pretrain_eval_mixed.py
│       ├── reasoning_eval.py
│       └── training_dynamics.py
│
├── finetune/
│   ├── __init__.py
│   ├── finetune.py
│   ├── finetune_colab.py
│   └── finetune_colab_mixed_data_size.py
│
├── reasoning/
│   ├── generate_reasoning_data.py
│   ├── generate_reasoning_data_v2.py
│   └── data/
│       ├── final_validation.json
│       └── statistics.json
│
├── tests/
│   ├── test_causal_mask.py
│   ├── test_checkpoint.py
│   ├── test_config.py
│   ├── test_full_dataset.py
│   ├── test_model.py
│   ├── test_real_data.py
│   ├── test_tiny_overfit.py
│   ├── test_training_smoke.py
│   └── test_validation.py
│
├── Tokenizer_Stats/
│   └── tokenizer statistics and frequency plots
│
├── DC_Tokenization.ipynb
├── train.py
├── train_colab.py
└── ...
```

Generated large datasets, checkpoints, and other large binary artifacts are not intended to be committed to Git.

---

## 2. External Artifacts

### Complete Nepali Code/data

[Click here- Nepali Code/data](https://drive.google.com/drive/folders/1b5zzcz529Pmc8J2bBJqJUleOulcBbKbM?usp=sharing)

### Nepali data

**Complete Nepali data:**

[Click here- Nepali dataset](https://drive.google.com/file/d/1T7VPZhrL5V6akaIqqz4Avo7RQswz3ZFS/view?usp=sharing)


**Nepali Tokenized corpus:**

[Click here- Nepali Tokenized Corpus](https://drive.google.com/drive/folders/1jvypa_oln2AZx4Zmg6mk85fKLyVhac1f?usp=sharing)

### Nepali tokenizer

**Nepali tokenizer JSON:**

[Click here- Nepali Tokenizer](https://drive.google.com/file/d/13afURtcjk1QJxqVbVf0GGRTI1avDL2oa/view?usp=sharing)

The final model uses the **8,000-token Nepali BPE tokenizer**.

### Pretrained model

**Nepali Phase 2 checkpoints (`.pt`, including intermediate checkpoints and final checkpoint):**

[Click here- Nepali Phase 2 Checkpoints](https://drive.google.com/drive/folders/1b3134dzSeUg59Cim00r1IAKMNsVWGNBG?usp=sharing)

### Fine-tuned model

**Nepali Phase 3 checkpoints (`.pt`, including intermediate checkpoints and final checkpoint):**

[Click here- Nepali Phase 3 Checkpoints](https://drive.google.com/drive/folders/18IZ4gl8UeJQgHMco0UPtQodgOsHwmuVd?usp=sharing)

### Colab-Run Notebooks

**Phase 1:**
[Click here- Phase1](https://colab.research.google.com/drive/1p8iMWX5fFZwa1Iy5Lvpg9PhLkfLpptZm?usp=sharing)

**Phase 2:**
[Click here- Phase2](https://colab.research.google.com/drive/15FSX-UsZad9EP5k-H0aN68G4ODklK0nt?usp=sharing)

**Phase 3:**
[Click here- Phase3](https://colab.research.google.com/drive/1hocG9BeC6CZ7eLS3Q6GX7v-r81U0uf1J?usp=sharing)

---

## 3. Requirements

The implementation uses Python and PyTorch.

Recommended environment:

- Python 3.10+
- PyTorch
- NumPy
- `tokenizers`
- `tqdm`
- Matplotlib
- SacreBLEU
- pandas (used by tokenizer-statistics notebooks/scripts)

Install the required runtime packages:

```bash
pip install torch numpy tokenizers tqdm matplotlib sacrebleu pandas
```

For GPU training, install a PyTorch build compatible with the CUDA version available on the machine/Colab runtime.

The model itself does **not** use:

- pretrained language models
- pretrained tokenizers
- HuggingFace Transformer model classes
- `nn.Transformer*`
- any pre-built attention block

The Transformer is implemented directly using basic PyTorch components.

---

# 4. Tokenization

## 4.1 Tokenizer used

The Nepali model uses a tokenizer trained from scratch with the Hugging Face `tokenizers` library:

- Algorithm: **BPE**
- Vocabulary size: **8,000**
- Pre-tokenizer: `Whitespace`
- Minimum token frequency: `2`
- Special tokens:
  - `<unk>`
  - `<pad>`
  - `<bos>`
  - `<eos>`

The tokenizer is independent of the other project language and is used unchanged during model training and reasoning fine-tuning.

## 4.2 Reproducing tokenizer training

Tokenizer preparation/training is documented in:

```text
DC_Tokenization.ipynb
```

The relevant tokenizer cells:

1. Create a plain-text tokenizer-training corpus from the deduplicated Nepali JSONL corpus.
2. Train the 8K BPE tokenizer.
3. Save the tokenizer JSON.
4. Run tokenizer statistics on the validation/held-out data.

The tokenizer-training text is produced by streaming the deduplicated corpus line-by-line, so the complete corpus does not need to be loaded into RAM.

The relevant input is:

```text
nepali_merged_deduplicated.jsonl
```

The tokenizer-training text is:

```text
nepali_500M_tokenizer_training.txt
```

The resulting tokenizer is:

```text
nepali_bpe_8000_sp.json
```

### Reproduction steps

Open `DC_Tokenization.ipynb` and set the project root to the location containing the Nepali data.

The important paths are:

```text
Nepali/data/cleaned/final/nepali_merged_deduplicated.jsonl
Nepali/data/tokenizer/
```

Then run the tokenizer preparation and BPE-training cells.

The BPE training logic is equivalent to:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
)

tokenizer.train(
    files=["nepali_500M_tokenizer_training.txt"],
    trainer=trainer,
)

tokenizer.save("nepali_bpe_8000_sp.json")
```

If the tokenizer JSON already exists, the notebook can load it instead of retraining it.

---

# 5. Tokenizing the Corpus

After the tokenizer is trained, the cleaned corpus is converted into `uint16` token-ID shards using:

```text
data/tokenize_corpus.py
```

Example:

```bash
python -m Nepali.data.tokenize_corpus \
    --input /path/to/nepali_500M_tokenizer_training.txt \
    --tokenizer /path/to/nepali_bpe_8000_sp.json \
    --output Nepali/data/tokenized
```

Default shard size:

```text
10,000,000 tokens
```

Output:

```text
Nepali/data/tokenized/
├── train_000.bin
├── train_001.bin
├── ...
└── metadata.json
```

Each `.bin` file stores token IDs as `uint16`.

---

# 6. Train / Validation / Test Split

The tokenized corpus is split with:

```text
data/prepare_validation.py
```

Default sizes:

```text
Training     : 500,000,000 tokens
Validation   : 10,000,000 tokens
Test         : 10,000,000 tokens
Shard size   : 10,000,000 tokens
```

Example:

```bash
python -m Nepali.data.prepare_validation \
    --input Nepali/data/tokenized \
    --train-tokens 500000000 \
    --validation-tokens 10000000 \
    --test-tokens 10000000 \
    --shard-size 10000000
```

If the split output already exists, use:

```bash
python -m Nepali.data.prepare_validation \
    --input Nepali/data/tokenized \
    --train-tokens 500000000 \
    --validation-tokens 10000000 \
    --test-tokens 10000000 \
    --shard-size 10000000 \
    --overwrite
```

The resulting layout is:

```text
Nepali/data/tokenized/
├── train_*.bin
├── validation/
│   └── val_*.bin
├── test/
│   └── test_*.bin
└── split_metadata.json
```

The validation/test ranges are taken after the training-token range so that the token positions used for these splits do not overlap with the training range.

---

# 7. Model Implementation

The model is a decoder-only causal Transformer implemented from scratch.

## Architecture

The implementation consists of:

```text
model/lm.py
    └── LanguageModel
        ├── token embedding
        ├── learned positional embedding
        ├── embedding dropout
        ├── Transformer stack
        │   └── TransformerBlock × N
        │       ├── causal multi-head self-attention
        │       ├── residual connection
        │       ├── layer normalization
        │       ├── feed-forward network
        │       └── residual connection
        ├── final normalization
        └── vocabulary projection
```

### Nepali configuration

Defined in:

```text
config/model_config.py
```

Current configuration:

```text
Vocabulary size       : 8,000
Model dimension       : 448
Number of layers      : 11
Number of heads       : 8
Feed-forward size     : 1,344
Maximum sequence      : 512
Dropout               : 0.1
Attention dropout     : 0.1
Embedding/output tie  : enabled
```

Each attention head has:

```text
448 / 8 = 56 dimensions
```

The model uses learned absolute positional embeddings.

The input embedding and output LM head are tied when `tie_embeddings=True`.

---

# 8. Pretraining

## Configuration

Pretraining defaults are defined in:

```text
config/train_config.py
```

Current defaults:

```text
Sequence length        : 512
Batch size             : 16
Maximum steps          : 61,036
Learning rate          : 2e-4
Minimum LR ratio       : 0.1
AdamW beta1            : 0.9
AdamW beta2            : 0.95
Weight decay           : 0.01
Warmup steps           : 2,000
Gradient clipping      : 1.0
Log interval           : 50 steps
Validation interval    : 2,000 steps
Validation batches     : 100
Checkpoint interval    : 2,000 steps
Seed                   : 42
Mixed precision        : enabled
```

## Local training

Run from the project root:

```bash
python -m Nepali.train
```

Optional overrides:

```bash
python -m Nepali.train \
    --train-dir Nepali/data/tokenized \
    --validation-dir Nepali/data/tokenized/validation \
    --checkpoint-dir Nepali/checkpoints \
    --max-steps 61036
```

## Colab GPU training

For the intended GPU/Colab workflow, use:

```text
train_colab.py
```

The launcher expects the project at:

```text
/content/drive/MyDrive/LMA_IP
```

and temporary training inputs at:

```text
/content/LMA_IP_DATA/train
/content/LMA_IP_DATA/validation
```

Persistent checkpoints are written to:

```text
/content/drive/MyDrive/LMA_IP/Nepali/checkpoints
```

Run:

```bash
python -m Nepali.train_colab
```

Short smoke run:

```bash
python -m Nepali.train_colab --max-steps 100
```

Resume:

```bash
python -m Nepali.train_colab \
    --resume /content/drive/MyDrive/LMA_IP/Nepali/checkpoints/step_XXXXXX.pt
```

The Colab launcher is a wrapper around the main `Nepali.train` pipeline; it does not define a second training implementation.

---

# 9. Checkpoints and Resume

Checkpoint saving is implemented in `train.py`.

A pretraining checkpoint contains:

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict
step
loss
train_config
model_config
Python RNG state
NumPy RNG state
PyTorch RNG state
CUDA RNG state (when available)
```

Therefore a training run can be resumed from an intermediate checkpoint rather than restarting from the beginning.

Checkpoint naming:

```text
step_XXXXXX.pt
final.pt
```

To resume:

```bash
python -m Nepali.train \
    --resume /path/to/step_XXXXXX.pt
```

The same resume mechanism is used for Phase 3 fine-tuning checkpoints.

---

# 10. Phase 2 Evaluation

## 10.1 Language-model evaluation

Run:

```text
eval/evaluate_lm.py
```

Example:

```bash
python -m Nepali.eval.evaluate_lm \
    --checkpoint-dir Nepali/checkpoints \
    --test-dir Nepali/data/tokenized/test \
    --test-text /path/to/nepali_test.jsonl \
    --output Nepali/results/lm_metrics.json
```

The evaluation computes:

- cross-entropy loss
- perplexity
- BPB

Evaluation uses contiguous, non-overlapping test blocks instead of creating one sliding-window sample for every token.

## 10.2 Text generation

Generation is implemented in:

```text
eval/generate.py
```

The default generation settings include:

```text
Greedy
Temperature 0.5
Temperature 1.0
Temperature 1.5
```

Example:

```bash
python -m Nepali.eval.generate \
    --checkpoint-dir Nepali/checkpoints \
    --tokenizer /path/to/nepali_bpe_8000_sp.json \
    --test-text /path/to/nepali_test.jsonl \
    --output Nepali/results/generation_samples.jsonl \
    --num-samples 100 \
    --prefix-tokens 64 \
    --generation-tokens 64 \
    --batch-size 8 \
    --seed 42
```

## 10.3 Generation metrics

Metrics are implemented in:

```text
eval/metrics.py
```

The script reports:

- BLEU-4
- chrF++
- ROUGE-L
- repetition rate
- Distinct-1
- Distinct-2

Run:

```bash
python -m Nepali.eval.metrics \
    --input Nepali/results/generation_samples.jsonl \
    --output Nepali/results/generation_metrics.json
```

---

# 11. Attention Evaluation

Attention extraction and heatmap generation are implemented in:

```text
eval/attention.py
```

Example:

```bash
python -m Nepali.eval.attention \
    --checkpoint-dir Nepali/checkpoints \
    --tokenizer /path/to/nepali_bpe_8000_sp.json \
    --test-text /path/to/nepali_test.jsonl \
    --output-dir Nepali/results/attention \
    --document-index 0 \
    --sequence-length 64 \
    --heads 0 1 2 3
```

The script produces:

```text
attention_<early/late>_layer_XX_head_XX.png
attention_entropy.json
attention_distance.json
attention_metadata.json
```

It computes:

- attention heatmaps
- per-head/layer attention entropy
- mean attention distance

The visualization code uses query position on the y-axis and key position on the x-axis.

Additional plotting helpers are available under:

```text
eval/
```

including:

```text
plot_attention_local_global.py
plot_attention_summary.py
```

---

# 12. Phase 3 Synthetic Reasoning Data

The reasoning dataset is generated locally; it does not require downloading an external reasoning benchmark.

The main generator is:

```text
reasoning/generate_reasoning_data_v2.py
```

Default sizes:

```text
Training     : 20,000
Validation   : 2,000
Test         : 3,000
```

Task families include:

```text
direct comparison
reverse comparison
equality
2-hop transitivity
3-hop transitivity
ordering / ranking
```

The generated prompts are in Nepali/Devanagari.

The generator uses fixed random seeds and maintains separate template-ID families for training and test examples. It also enforces global prompt uniqueness.

Generate the dataset with:

```bash
python -m Nepali.reasoning.generate_reasoning_data_v2 \
    --output-dir Nepali/reasoning/data \
    --seed 42
```

If the script exposes explicit size arguments in the installed version, they can be used to override the default train/validation/test sizes.

The generated directory should contain:

```text
Nepali/reasoning/data/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── statistics.json
└── ...
```

---

# 13. Phase 3 Reasoning Fine-tuning

## 13.1 Phase 3 Experiment Used for Submission

Phase 3 supports both the standard single-data-size fine-tuning pipeline and the mixed-data-size experiment.

For the submitted Phase 3 results, the **Mixed Data Size Experiment** is used as the final Nepali experiment. It is implemented in:

```text
finetune/finetune_colab_mixed_data_size.py
```

The corresponding evaluation utilities are:

```text
eval/finetune/attention_compare_mixed.py
eval/finetune/pretrain_eval_mixed.py
eval/finetune/training_dynamics.py
```

The mixed-data experiment performs fine-tuning using the configured reasoning-data sizes while keeping the pretrained Nepali model architecture and tokenizer fixed. The resulting runs are saved as resume-capable checkpoints.

The standard single-data-size fine-tuning implementation is also provided and remains runnable:

```text
finetune/finetune.py
finetune/finetune_colab.py
```

However, the single-data-size run is not the experiment used for the submitted Phase 3 results.

## 13.2 Standard Fine-tuning Pipeline

Fine-tuning starts from the Phase 2 Nepali pretrained checkpoint.

The tokenizer and vocabulary remain fixed.

### Colab workflow

For the standard single-data-size fine-tuning pipeline, use:

```text
finetune/finetune_colab.py
```

Expected local Colab data:

```text
/content/LMA_IP_DATA/reasoning/
├── train.jsonl
└── val.jsonl
```

Example:

```bash
python -m Nepali.finetune.finetune_colab \
    --model-path /path/to/pretrained_final.pt \
    --tokenizer /path/to/nepali_bpe_8000_sp.json \
    --data-dir /content/LMA_IP_DATA/reasoning \
    --output-dir /content/drive/MyDrive/LMA_IP/Nepali/finetune \
    --max-steps 2000 \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --seed 42
```

To resume Phase 3:

```bash
python -m Nepali.finetune.finetune_colab \
    --resume /path/to/phase3_step_XXXXXX.pt \
    --tokenizer /path/to/nepali_bpe_8000_sp.json \
    --data-dir /content/LMA_IP_DATA/reasoning \
    --output-dir /content/drive/MyDrive/LMA_IP/Nepali/finetune \
    --seed 42
```

The underlying implementation is:

```text
finetune/finetune.py
```

The fine-tuning objective applies the causal-LM loss to the answer portion of each reasoning example while prompt tokens are ignored.

Default Phase 3 settings include:

```text
Batch size             : 16
Maximum steps          : 2,000
Learning rate          : 5e-5
Weight decay           : 0.01
AdamW beta1            : 0.9
AdamW beta2            : 0.95
Gradient clipping      : 1.0
Warmup steps           : 100
Minimum LR ratio       : 0.1
Log interval           : 50
Validation interval    : 250
Checkpoint interval    : 250
Validation batches     : 100
Seed                   : 42
```

## 13.3 Mixed Data Size Experiment

The submitted Phase 3 experiment uses:

```text
finetune/finetune_colab_mixed_data_size.py
```

This pipeline performs fine-tuning with the configured reasoning-data sizes using the same pretrained Nepali model architecture and fixed Nepali tokenizer/vocabulary.

The different data-size runs are kept separately so that their checkpoints and training histories can be evaluated independently.

The corresponding mixed-data evaluation utilities are:

```text
eval/finetune/pretrain_eval_mixed.py
eval/finetune/attention_compare_mixed.py
eval/finetune/training_dynamics.py
```

The mixed-data experiment is the source of the Phase 3 results used for submission. Its checkpoints are available through the **Phase 3 Mixed Data Checkpoints** Drive link in Section 2.

The standard single-data-size implementation remains available for reproducibility and testing, but its checkpoint is not used as the submitted Phase 3 model.

---

# 14. Phase 3 Evaluation

## 14.1 Reasoning evaluation

Implemented in:

```text
eval/finetune/reasoning_eval.py
```

It compares the pretrained and fine-tuned checkpoints on the held-out reasoning test set.

Example:

```bash
python -m Nepali.eval.finetune.reasoning_eval \
    --pretrained /path/to/pretrained_final.pt \
    --finetuned /path/to/finetuned_final.pt \
    --tokenizer /path/to/nepali_bpe_8000_sp.json \
    --test Nepali/reasoning/data/test.jsonl \
    --output-dir Nepali/results/finetune/reasoning \
    --seed 42
```

## 14.2 Pretrained vs. Fine-tuned Attention

Implemented in:

```text
eval/finetune/attention_compare.py
```

Example:

```bash
python -m Nepali.eval.finetune.attention_compare \
    --pretrained /path/to/pretrained_final.pt \
    --finetuned /path/to/finetuned_final.pt \
    --tokenizer /path/to/nepali_bpe_8000_sp.json \
    --test Nepali/reasoning/data/test.jsonl \
    --output-dir Nepali/results/finetune/attention \
    --num-prompts 50 \
    --sequence-length 96
```

This produces representative early/late attention heatmaps and machine-readable attention entropy/distance comparisons.

## 14.3 Mixed Data Size Evaluation

The submitted Phase 3 experiment uses the mixed-data-size evaluation utilities:

```text
eval/finetune/pretrain_eval_mixed.py
eval/finetune/attention_compare_mixed.py
eval/finetune/training_dynamics.py
```

Training dynamics can be generated with:

```bash
python -m Nepali.eval.finetune.training_dynamics \
    --finetune-root Nepali/finetune/mixed_data_size \
    --output-root Nepali/results/finetune
```

The script reads the saved mixed-data experiment checkpoints/training histories and generates:

- training loss curves
- validation loss curves
- training/validation perplexity curves
- learning-rate curves
- machine-readable training histories

The mixed-data attention comparison is implemented separately in:

```text
eval/finetune/attention_compare_mixed.py
```

and the corresponding pretrained evaluation utilities are implemented in:

```text
eval/finetune/pretrain_eval_mixed.py
```

---

# 15. Reproducibility Checklist

To reproduce the Nepali pipeline from the supplied artifacts:

### Step 1 — Environment

```bash
pip install torch numpy tokenizers tqdm matplotlib sacrebleu pandas
```

Use a CUDA-enabled PyTorch installation for GPU training and generation.

### Step 2 — Obtain artifacts

Download the required files from the links at the top of this README:

```text
DATA
TOKENIZER
PRETRAINED CHECKPOINTS
PHASE 3 MIXED DATA CHECKPOINTS
REASONING DATA
```

### Step 3 — Tokenizer

If reproducing the tokenizer rather than using the supplied tokenizer:

1. Obtain the deduplicated/normalized Nepali corpus.
2. Run the tokenizer-preparation cells in `DC_Tokenization.ipynb`.
3. Train the 8K BPE tokenizer.
4. Save the tokenizer JSON.
5. Run the tokenizer-statistics cells if the tokenizer statistics need to be regenerated.

### Step 4 — Tokenize

```bash
python -m Nepali.data.tokenize_corpus \
    --input /path/to/nepali_500M_tokenizer_training.txt \
    --tokenizer /path/to/nepali_bpe_8000_sp.json \
    --output Nepali/data/tokenized
```

### Step 5 — Split

```bash
python -m Nepali.data.prepare_validation \
    --input Nepali/data/tokenized \
    --train-tokens 500000000 \
    --validation-tokens 10000000 \
    --test-tokens 10000000 \
    --shard-size 10000000
```

### Step 6 — Pretrain

For a GPU/Colab run:

```bash
python -m Nepali.train_colab
```

For a local run:

```bash
python -m Nepali.train
```

Use `--resume` with an intermediate `.pt` checkpoint if the run is interrupted.

### Step 7 — Phase 2 evaluation

Run:

```bash
python -m Nepali.eval.evaluate_lm ...
python -m Nepali.eval.generate ...
python -m Nepali.eval.metrics ...
python -m Nepali.eval.attention ...
```

using the exact paths to the downloaded checkpoint, tokenizer, and held-out data.

### Step 8 — Generate reasoning data

```bash
python -m Nepali.reasoning.generate_reasoning_data_v2 \
    --output-dir Nepali/reasoning/data \
    --seed 42
```

### Step 9 — Phase 3 Mixed Data Fine-tuning

For the submitted Phase 3 experiment, use:

```text
finetune/finetune_colab_mixed_data_size.py
```

Start from the Phase 2 Nepali pretrained checkpoint and use the Nepali tokenizer unchanged.

The mixed-data experiment produces separate checkpoint and training-history outputs for the configured reasoning-data sizes.

The resulting checkpoints are stored in the **Phase 3 Mixed Data Checkpoints** Drive folder linked in Section 2.

The standard single-data-size fine-tuning pipeline is also available:

```text
finetune/finetune.py
finetune/finetune_colab.py
```

but it is not the pipeline used for the submitted Phase 3 results.

### Step 10 — Phase 3 evaluation

For the mixed-data experiment, use:

```text
eval/finetune/pretrain_eval_mixed.py
eval/finetune/attention_compare_mixed.py
eval/finetune/training_dynamics.py
```

For the standard single-data-size pipeline, the corresponding utilities are also available under:

```text
eval/finetune/
```

---

# 16. Testing

Tests are located in:

```text
tests/
```

Run the complete test suite from the project root:

```bash
pytest Nepali/tests
```

The tests cover configuration validation, model behavior, causal masking, checkpoint handling, dataset behavior, validation, tiny-model overfitting, and training smoke tests.

A shorter model/masking check can be run with:

```bash
pytest Nepali/tests/test_model.py Nepali/tests/test_causal_mask.py
```

Tests that require real dataset/checkpoint artifacts should be run after the corresponding Drive artifacts have been downloaded.

---

# 17. Important Path Conventions

The code supports explicit paths through command-line arguments for the major pipelines.

For Colab training, the intended layout is:

```text
/content/drive/MyDrive/LMA_IP/
└── Nepali/
    ├── checkpoints/
    ├── finetune/
    └── ...

/content/LMA_IP_DATA/
├── train/
├── validation/
└── reasoning/
```

For the mixed-data fine-tuning experiment, the relevant output directory is:

```text
Nepali/finetune/mixed_data_size/
```

For a normal local checkout, paths can instead be supplied directly through the relevant command-line arguments.

---

# 18. Reproducibility Controls

The main training and generation pipelines use seed `42` by default.

The code explicitly seeds:

```text
Python random
NumPy
PyTorch
CUDA RNGs when CUDA is available
```

Training checkpoints additionally store RNG states so an interrupted training run can restore the stochastic training state.

The reasoning-data generator also accepts a seed and generates deterministic data for a given seed/configuration.

For comparable reruns, keep the following unchanged:

```text
Python/PyTorch environment
model_config.py
train_config.py
tokenizer JSON
input dataset/splits
random seed
batch size
sequence length
maximum training steps
optimizer hyperparameters
learning-rate schedule
checkpoint/resume state
```

For the mixed-data Phase 3 experiment, also keep the configured reasoning-data sizes and corresponding split files unchanged.

---

# 19. Code-to-Task Map

| Task | Main code |
|---|---|
| Tokenizer corpus preparation | `DC_Tokenization.ipynb` |
| BPE tokenizer training | `DC_Tokenization.ipynb` |
| Tokenization | `data/tokenize_corpus.py` |
| Dataset split | `data/prepare_validation.py` |
| Training dataset | `data/dataset.py` |
| Validation dataset | `data/validation_dataset.py` |
| Model configuration | `config/model_config.py` |
| Training configuration | `config/train_config.py` |
| Token embedding / LM | `model/lm.py` |
| Transformer | `model/transformer.py` |
| Transformer block | `model/transformer_block.py` |
| Attention | `model/attention.py` |
| Positional embedding | `model/positional_embedding.py` |
| Pretraining | `train.py` |
| Colab pretraining | `train_colab.py` |
| LM evaluation | `eval/evaluate_lm.py` |
| Generation | `eval/generate.py` |
| Generation metrics | `eval/metrics.py` |
| Attention extraction | `eval/attention.py` |
| Reasoning data generation | `reasoning/generate_reasoning_data_v2.py` |
| Standard reasoning fine-tuning | `finetune/finetune.py` |
| Standard Colab fine-tuning | `finetune/finetune_colab.py` |
| Mixed-data-size fine-tuning | `finetune/finetune_colab_mixed_data_size.py` |
| Standard reasoning evaluation | `eval/finetune/reasoning_eval.py` |
| Mixed-data pretrained evaluation | `eval/finetune/pretrain_eval_mixed.py` |
| Standard attention comparison | `eval/finetune/attention_compare.py` |
| Mixed-data attention comparison | `eval/finetune/attention_compare_mixed.py` |
| Fine-tuning dynamics | `eval/finetune/training_dynamics.py` |
| Tests | `tests/` |

---


## 20. Artifact Summary

The repository contains the code and lightweight metadata required to reproduce the Nepali pipeline. Large binary artifacts are provided separately through Google Drive.

| Artifact | Location / placeholder |
|---|---|
| Cleaned/deduplicated corpus | https://drive.google.com/file/d/1T7VPZhrL5V6akaIqqz4Avo7RQswz3ZFS/view?usp=sharing |
| Tokenized corpus | https://drive.google.com/drive/folders/1jvypa_oln2AZx4Zmg6mk85fKLyVhac1f?usp=sharing |
| 8K tokenizer JSON | https://drive.google.com/file/d/13afURtcjk1QJxqVbVf0GGRTI1avDL2oa/view?usp=sharing |
| Phase 2 checkpoints | https://drive.google.com/drive/folders/1b3134dzSeUg59Cim00r1IAKMNsVWGNBG?usp=sharing |
| Phase 3 checkpoints | https://drive.google.com/drive/folders/18IZ4gl8UeJQgHMco0UPtQodgOsHwmuVd?usp=sharing |
| Reasoning dataset | https://drive.google.com/drive/folders/1pvewZ69xPaQGW9VMicC5krZYdtYnI5X6?usp=sharing |

No dataset-collection reproduction procedure is included here; the README starts from the supplied cleaned/deduplicated corpus for the tokenizer/data pipeline. Tokenizer training itself is reproducible from that corpus using `DC_Tokenization.ipynb`.
