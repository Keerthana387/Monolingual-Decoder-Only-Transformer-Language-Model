# Telugu Monolingual Transformer Language Model

This directory contains the Telugu implementation for the LMA Individual Project. It includes the Telugu tokenizer pipeline, tokenized-corpus preparation, decoder-only Transformer implementation, pretraining pipeline, Phase 2 evaluation, synthetic reasoning-data generation, Phase 3 reasoning fine-tuning, mixed-data-size experiments, and post-fine-tuning evaluation.

> **Scope:** This README documents the code, file organization, dependencies, inputs/outputs, and reproduction procedure. Results and interpretation are kept in the report.

---

## 1. Directory Structure

```text
Telugu/
├── config/
│   ├── __init__.py
│   ├── model_config.py
│   └── train_config.py
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── prepare_test.py
│   ├── prepare_validation.py
│   ├── test_dataset.py
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
│   ├── __init__.py
│   ├── attention.py
│   ├── common.py
│   ├── evaluate_lm.py
│   ├── generate.py
│   ├── make_generation_samples.py
│   ├── metrics.py
│   ├── plot_attention_local_global.py
│   ├── plot_attention_summary.py
│   ├── plot_generation_metrics.py
│   ├── plot_loss.py
│   ├── plot_lr_schedule.py
│   ├── plot_perplexity.py
│   ├── plot_ppl.py
│   └── finetune/
│       ├── attention_compare.py
│       ├── attention_compare_mixed.py
│       ├── pretrain_eval_mixed.py
│       ├── reasoning_eval.py
│       ├── reasoning_eval_mixed.py
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
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test.jsonl
│       └── statistics.json
│
├── tests/
│   ├── __init__.py
│   ├── test_casual_mask.py
│   ├── test_checkpoint.py
│   ├── test_full_dataset.py
│   ├── test_model.py
│   ├── test_real_data.py
│   ├── test_tiny_overfit.py
│   ├── test_training_smoke.py
│   └── test_validation.py
│
├── Tokenizer_Stats/
│   ├── tokenizer frequency CSV files
│   ├── top-token frequency plots
│   └── validation/source-wise statistics
│
├── Bonus/
│   ├── config/
│   ├── model/
│   ├── eval/
│   ├── results/
│   ├── train.py
│   ├── train_colab.py
│   └── training_output.txt
│
├── DC_Tokenization.ipynb
├── train.py
├── train_colab.py
├── training_output.txt
├── phase2_report.md
└── results/
    ├── attention/
    ├── finetune/
    └── generation/
```

Generated large datasets, tokenized corpora, checkpoints, and other large binary artifacts are not intended to be committed to Git.

---

## 2. External Artifacts

### Complete Telugu Code/data

[Click here- Telugu Code/data](https://drive.google.com/drive/folders/1LG5Jvhv6yG9cm47kxqPne1EHVR_bqnut?usp=sharing)

### Telugu data

**Complete Telugu dataset:**

[Click here- Telugu dataset](https://drive.google.com/file/d/1Flqz6PAJKRcpvZ-KrJ8YfodC4f84TUeh/view?usp=sharing)

**Telugu tokenized corpus:**

[Click here- Telugu Tokenized Corpus](https://drive.google.com/drive/folders/1o-rFGvDB4n7fOVreHNbRyMR_EpreEiPP?usp=sharing)

### Telugu tokenizer

**Telugu SentencePiece tokenizer (`.model` and `.vocab`):**

[Click here- Telugu Tokenizer](https://drive.google.com/drive/folders/1PKsO7cOgUkQpTHNoZkM29e9JM6KwkxYM?usp=sharing)

The final model uses the **6,000-token Telugu BPE tokenizer**.

### Pretrained model

**Telugu Phase 2 checkpoints (`.pt`, including intermediate checkpoints and final checkpoint):**

[Click here- Telugu Phase 2 Checkpoints](https://drive.google.com/drive/folders/1L2WkWaXQdpNuY563F48_AXDlnKGO_nNy?usp=sharing)

### Phase 3 mixed-data model

**Telugu Phase 3 mixed-data checkpoints (`.pt`, including all experiment checkpoints):**

[Click here- Telugu Phase 3 Checkpoints](https://drive.google.com/drive/folders/1EeQTjbE_Walg6bSMxpxZBO3DHp3iacrC?usp=sharing)

The submitted Phase 3 experiment is the **mixed-data-size experiment** implemented by:

```text
finetune/finetune_colab_mixed_data_size.py
```

### Reasoning data

**Telugu synthetic reasoning dataset:**

[Click here- Telugu Reasonng Dataset](https://drive.google.com/drive/folders/1cT1fxocvkX6wnJXCB5gbUkGPsTUOTuYV?usp=sharing)

### Colab-Run Notebooks

**Phase 1:**
[Click here- Phase1](https://colab.research.google.com/drive/13AM39r_rOlICN8DYgdndXS1fuaXbzgCM?usp=sharing)

**Phase 2:**
[Click here- Phase2](https://colab.research.google.com/drive/1yB1zHaq-fbpLcQG0056HjCoUQ4rKdOc_?usp=sharing)

**Phase 3:**
[Click here- Phase3](https://colab.research.google.com/drive/1i4ucOwMcl-CrKi5cOZhtDbor643LZlx8?usp=sharing)

**Bonus:**
[Click here- Bonus](https://colab.research.google.com/drive/1wvO0Z1GclQitrsGp2aQWoCdLRYXqDZdD?usp=sharing)

---

## 3. Requirements

The implementation uses Python and PyTorch.

Recommended environment:

- Python 3.10+
- PyTorch
- NumPy
- `sentencepiece`
- `tqdm`
- Matplotlib
- SacreBLEU
- pandas

Install the required runtime packages:

```bash
pip install torch numpy sentencepiece tqdm matplotlib sacrebleu pandas
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

The Telugu model uses a tokenizer trained from scratch with **SentencePiece BPE**:

- Algorithm: **BPE**
- Final vocabulary size: **6,000**
- Character coverage: `1.0`
- Training input: complete tokenizer-training text
- `input_sentence_size`: `0`
- `shuffle_input_sentence`: enabled
- `hard_vocab_limit`: disabled
- Training threads: `4`
- Special token IDs:
  - `<unk>` → `0`
  - `<pad>` → `1`
  - `<bos>` → `2`
  - `<eos>` → `3`

The tokenizer is independent of the other project language and is used unchanged during model training and Phase 3 reasoning fine-tuning.

Candidate vocabularies were trained for:

```text
5,000
6,000
7,000
8,000
9,000
10,000
```

The tokenizer files for the final model are:

```text
telugu_bpe_6000.model
telugu_bpe_6000.vocab
```

## 4.2 Reproducing tokenizer training

Tokenizer preparation/training is documented in:

```text
DC_Tokenization.ipynb
```

The relevant tokenizer cells:

1. Load the selected/deduplicated Telugu corpus.
2. Create the tokenizer-training text by streaming the corpus.
3. Train SentencePiece BPE tokenizers for the candidate vocabulary sizes.
4. Load the trained tokenizer models.
5. Run tokenizer statistics on validation/held-out data.
6. Use the 6K tokenizer for the final Telugu model.

The tokenizer-training text is created from:

```text
telugu_merged_deduplicated.jsonl
```

and saved as:

```text
telugu_500M_tokenizer_training.txt
```

The intended Drive paths used during preparation are:

```text
/content/drive/MyDrive/LMA_IP/Telugu/data/cleaned/final/telugu_merged_deduplicated.jsonl
/content/drive/MyDrive/LMA_IP/Telugu/data/tokenizer/
```

### Reproduction steps

Open `DC_Tokenization.ipynb` and set the project paths to the location of the prepared Telugu corpus.

The important paths are:

```text
Telugu/data/cleaned/final/telugu_merged_deduplicated.jsonl
Telugu/data/tokenizer/
```

The notebook creates:

```text
Telugu/data/tokenizer/telugu_500M_tokenizer_training.txt
```

The SentencePiece training configuration is equivalent to:

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="telugu_500M_tokenizer_training.txt",
    model_prefix="telugu_bpe_6000",
    model_type="bpe",
    vocab_size=6000,
    character_coverage=1.0,
    input_sentence_size=0,
    shuffle_input_sentence=True,
    hard_vocab_limit=False,
    unk_id=0,
    pad_id=1,
    bos_id=2,
    eos_id=3,
    num_threads=4,
)
```

This produces:

```text
telugu_bpe_6000.model
telugu_bpe_6000.vocab
```

If the tokenizer files already exist, they can be loaded directly instead of retraining them.

The tokenizer-statistics outputs are stored under:

```text
Tokenizer_Stats/
```

and include token-frequency tables, top-token frequency plots, and validation/source-wise statistics for the candidate vocabulary sizes.

---

# 5. Tokenizing the Corpus

After the tokenizer is trained, the corpus is converted into `uint16` token-ID shards using:

```text
data/tokenize_corpus.py
```

Example:

```bash
python -m Telugu.data.tokenize_corpus \
    --input /path/to/telugu_500M_tokenizer_training.txt \
    --tokenizer /path/to/telugu_bpe_6000.model \
    --output Telugu/data/tokenized
```

The tokenization script processes the corpus incrementally rather than loading the entire corpus into RAM.

Default shard size:

```text
10,000,000 tokens
```

Typical output:

```text
Telugu/data/tokenized/
├── train_000.bin
├── train_001.bin
├── ...
└── metadata.json
```

Each `.bin` file stores token IDs as `uint16`.

---

# 6. Train / Validation / Test Split

The tokenized corpus is split using:

```text
data/prepare_validation.py
data/prepare_test.py
```

The validation and test preparation scripts operate on the tokenized corpus and create separate binary shards for evaluation.

Typical layout:

```text
Telugu/data/tokenized/
├── train_*.bin
├── validation/
│   └── val_*.bin
├── test/
│   └── test_*.bin
└── split_metadata.json
```

For a token-count-based split, use:

```bash
python -m Telugu.data.prepare_validation \
    --input Telugu/data/tokenized \
    --train-tokens 500000000 \
    --validation-tokens 10000000 \
    --test-tokens 10000000 \
    --shard-size 10000000
```

If the split needs to be regenerated:

```bash
python -m Telugu.data.prepare_validation \
    --input Telugu/data/tokenized \
    --train-tokens 500000000 \
    --validation-tokens 10000000 \
    --test-tokens 10000000 \
    --shard-size 10000000 \
    --overwrite
```

`prepare_test.py` is also provided for the final test-data preparation used by the evaluation pipeline.

The resulting validation/test data is kept separate from the training shard range.

---

# 7. Model Implementation

The model is a decoder-only causal Transformer implemented from scratch.

## Architecture

The implementation consists of:

```text
model/lm.py
    └── TeluguLanguageModel
        ├── token embedding
        ├── learned positional embedding
        ├── embedding dropout
        ├── Transformer stack
        │   └── TransformerBlock × 10
        │       ├── causal multi-head self-attention
        │       ├── residual connection
        │       ├── layer normalization
        │       ├── feed-forward network
        │       └── residual connection
        ├── final normalization
        └── tied vocabulary projection
```

### Telugu configuration

Defined in:

```text
config/model_config.py
```

Current configuration:

```text
Vocabulary size       : 6,000
Model dimension       : 448
Number of layers      : 10
Number of heads       : 8
Feed-forward size     : 1,344
Maximum sequence      : 512
Dropout               : 0.1
Attention dropout     : 0.1
Embedding/output tie  : enabled
Padding token ID      : 0
```

Each attention head has:

```text
448 / 8 = 56 dimensions
```

The model uses learned absolute positional embeddings.

The input token embedding and output LM head are tied.

The resulting model contains:

```text
23,042,432 trainable parameters
```

## Model files

### `model/attention.py`

Implements multi-head causal self-attention:

1. Project input to Q, K and V.
2. Reshape into multiple attention heads.
3. Compute scaled dot-product attention.
4. Apply the causal mask.
5. Apply softmax and attention dropout.
6. Combine attention heads.
7. Apply the output projection.

Tensor layout:

```text
(B, T, 448)
      ↓
Q, K, V
      ↓
(B, 8, T, 56)
      ↓
attention scores
(B, 8, T, T)
      ↓
causal mask + softmax
      ↓
weighted V
      ↓
(B, T, 448)
```

### `model/positional_embedding.py`

Implements learned absolute positional embeddings for the 512-token context.

### `model/transformer_block.py`

Each Transformer block contains:

```text
LayerNorm
→ causal self-attention
→ residual connection

LayerNorm
→ feed-forward network
→ residual connection
```

The feed-forward network is:

```text
448 → 1344 → 448
```

### `model/transformer.py`

Builds the stack of 10 decoder-only Transformer blocks and applies the final normalization.

### `model/lm.py`

Defines `TeluguLanguageModel` and provides:

- token embedding;
- Transformer forward pass;
- vocabulary logits;
- causal next-token loss;
- tied input/output embeddings.

---

# 8. Pretraining

## Configuration

Pretraining defaults are defined in:

```text
config/train_config.py
```

Current configuration:

```text
Sequence length        : 512
Batch size             : 8
Maximum steps          : 100,000
Learning rate          : 3e-4
Minimum LR ratio       : 0.1
AdamW beta1            : 0.9
AdamW beta2            : 0.95
Weight decay           : 0.01
Warmup steps           : 2,000
Gradient clipping      : 1.0
Log interval           : 50 steps
Validation interval    : 1,000 steps
Validation batches     : 100
Checkpoint interval    : 1,000 steps
Seed                   : 42
DataLoader workers     : 0
Pin memory             : disabled
Mixed precision        : enabled
```

The recorded Colab run can apply its run-specific settings through `train_colab.py`.

## Local training

Run from the project root:

```bash
python -m Telugu.train
```

Optional overrides:

```bash
python -m Telugu.train \
    --train-dir Telugu/data/tokenized \
    --validation-dir Telugu/data/tokenized/validation \
    --checkpoint-dir Telugu/checkpoints \
    --max-steps 100000
```

## Colab GPU training

For the intended GPU/Colab workflow, use:

```text
train_colab.py
```

The training data is copied to local Colab runtime storage:

```text
/content/LMA_IP_DATA/train
/content/LMA_IP_DATA/validation
```

Persistent checkpoints are stored on Google Drive:

```text
/content/drive/MyDrive/LMA_IP/Telugu/checkpoints
```

Run:

```bash
python -m Telugu.train_colab
```

For a short smoke run:

```bash
python -m Telugu.train_colab --max-steps 100
```

Resume from a checkpoint:

```bash
python -m Telugu.train_colab \
    --resume /content/drive/MyDrive/LMA_IP/Telugu/checkpoints/step_XXXXXX.pt
```

The Colab launcher uses the same core training implementation rather than maintaining a separate model implementation.

---

# 9. Checkpoints and Resume

Checkpoint saving is implemented in `train.py`.

A pretraining checkpoint stores the training state required for recovery, including:

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict
current step
loss
training configuration
model configuration
random-state information
```

Checkpoint naming:

```text
step_XXXXXX.pt
final.pt
```

Resume:

```bash
python -m Telugu.train \
    --resume /path/to/step_XXXXXX.pt
```

The same checkpoint-based approach is used for Phase 3 fine-tuning.

---

# 10. Phase 2 Evaluation

## 10.1 Language-model evaluation

Run:

```text
eval/evaluate_lm.py
```

Example:

```bash
python -m Telugu.eval.evaluate_lm \
    --checkpoint-dir Telugu/checkpoints \
    --test-dir Telugu/data/tokenized/test \
    --test-text /path/to/telugu_test.jsonl \
    --output Telugu/results/lm_metrics.json
```

The evaluation computes:

- cross-entropy loss;
- perplexity;
- BPB.

The evaluation scripts operate on the held-out data and write machine-readable metric files.

## 10.2 Text generation

Generation is implemented in:

```text
eval/generate.py
```

The generation suite supports:

```text
Greedy decoding
Temperature 0.5
Temperature 1.0
Temperature 1.5
```

Example:

```bash
python -m Telugu.eval.generate \
    --checkpoint-dir Telugu/checkpoints \
    --tokenizer /path/to/telugu_bpe_6000.model \
    --test-text /path/to/telugu_test.jsonl \
    --output Telugu/results/generations.jsonl \
    --num-samples 100 \
    --prefix-tokens 64 \
    --generation-tokens 64 \
    --batch-size 8 \
    --seed 42
```

`make_generation_samples.py` can be used to create the stored generation sample tables used by the evaluation outputs.

## 10.3 Generation metrics

Metrics are implemented in:

```text
eval/metrics.py
```

The evaluation reports:

- BLEU-4;
- chrF++;
- ROUGE-L;
- repetition rate;
- Distinct-1;
- Distinct-2.

Example:

```bash
python -m Telugu.eval.metrics \
    --input Telugu/results/generations.jsonl \
    --output Telugu/results/generation_metrics.json
```

---

# 11. Attention Evaluation

Attention extraction is implemented in:

```text
eval/attention.py
```

Example:

```bash
python -m Telugu.eval.attention \
    --checkpoint-dir Telugu/checkpoints \
    --tokenizer /path/to/telugu_bpe_6000.model \
    --test-text /path/to/telugu_test.jsonl \
    --output-dir Telugu/results/attention \
    --document-index 0 \
    --sequence-length 64 \
    --heads 0 1 2 3
```

The attention toolkit produces:

```text
attention_<early/late>_layer_XX_head_XX.png
attention_entropy.json
attention_distance.json
attention_metadata.json
```

It computes:

- attention heatmaps;
- per-head/layer attention entropy;
- mean attention distance.

The visualization uses query position and key position as the attention-map axes.

Additional plotting utilities are available under:

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

The reasoning dataset is generated locally and is not dependent on downloading an external reasoning benchmark.

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

The generator contains Telugu-language templates for comparative and basic reasoning tasks, including:

```text
direct comparison
reverse comparison
equality
2-hop transitivity
3-hop transitivity
ordering / ranking
mixed transitive reasoning
distractor reasoning
mixed-question variants
```

The generator uses a fixed seed by default and validates the generated dataset.

The generated directory is:

```text
Telugu/reasoning/data/
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── statistics.json
```

Generate the dataset with:

```bash
python -m Telugu.reasoning.generate_reasoning_data_v2 \
    --output-dir Telugu/reasoning/data \
    --seed 42
```

For a smaller test generation:

```bash
python -m Telugu.reasoning.generate_reasoning_data_v2 \
    --output-dir Telugu/reasoning/test_data \
    --train-size 30 \
    --val-size 10 \
    --test-size 10 \
    --seed 42
```

The original reasoning dataset files are not modified by the mixed-data-size fine-tuning pipeline.

---

# 13. Phase 3 Reasoning Fine-tuning

## 13.1 Phase 3 Experiment Used for Submission

Phase 3 contains both:

1. the standard single-data-size fine-tuning implementation; and
2. the mixed-data-size experiment.

The **Mixed Data Size Experiment is the experiment used for the submitted Telugu Phase 3 results.**

It is implemented in:

```text
finetune/finetune_colab_mixed_data_size.py
```

The mixed-data experiment keeps the total training set fixed at:

```text
20,000 examples per run
```

and changes the composition between synthetic reasoning examples and real Telugu pretraining-corpus examples.

The six configured runs are:

```text
2,000 reasoning  + 18,000 real
5,000 reasoning  + 15,000 real
8,000 reasoning  + 12,000 real
12,000 reasoning +  8,000 real
16,000 reasoning +  4,000 real
20,000 reasoning +      0 real
```

Each run:

- starts from the same Phase 2 Telugu pretrained checkpoint;
- uses the same Telugu tokenizer;
- uses the same Telugu vocabulary;
- uses exactly 20,000 training examples;
- uses a fixed validation set;
- samples each selected training example at most once within a run;
- does not use the Phase 3 test set for training;
- uses only Telugu data;
- creates its own checkpoint/output directory.

The mixed-data script can consume the Phase 2 real training corpus in:

```text
.bin
.txt
.jsonl
```

format.

For `.bin` shards, token IDs are decoded using the same Phase 2 SentencePiece tokenizer before being converted into prompt/answer continuation examples.

For real-text examples:

```text
first 256 tokens → prompt
next 255 tokens   → answer
```

with the remaining sequence position reserved for EOS within the 512-token model context.

## 13.2 Mixed-data paths

Default reasoning data:

```text
/content/LMA_IP_DATA/reasoning/
```

Default Phase 2 real training corpus:

```text
/content/LMA_IP_DATA/train/
```

Default output:

```text
/content/drive/MyDrive/LMA_IP/Telugu/finetune/mixed_data_size/
```

## 13.3 Mixed-data configuration

The launcher defaults are:

```text
Total examples per run : 20,000
Batch size             : 16
Learning rate          : 5e-5
Seed                   : 42
Maximum sequence       : 512
Real prompt tokens     : 256
Real answer tokens     : 255

Reasoning sizes:
    2,000
    5,000
    8,000
    12,000
    16,000
    20,000
```

Since each run has exactly 20,000 examples and uses batch size 16:

```text
20,000 / 16 = 1,250 optimizer steps
```

Thus each mixed-data run performs one complete pass over its 20,000-example training set.

The underlying training implementation is:

```text
finetune/finetune.py
```

with the mixed-data launcher preparing the appropriate `train.jsonl` and fixed validation data for each run.

## 13.4 Run the submitted Phase 3 experiment

Start from the Phase 2 pretrained checkpoint:

```bash
python -m Telugu.finetune.finetune_colab_mixed_data_size \
    --model-path /path/to/Telugu/pretrained/final.pt \
    --tokenizer /path/to/telugu_bpe_6000.model \
    --reasoning-dir /content/LMA_IP_DATA/reasoning \
    --pretrain-data-dir /content/LMA_IP_DATA/train \
    --output-base /content/drive/MyDrive/LMA_IP/Telugu/finetune/mixed_data_size \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --seed 42
```

To run only selected mixtures:

```bash
python -m Telugu.finetune.finetune_colab_mixed_data_size \
    --model-path /path/to/Telugu/pretrained/final.pt \
    --tokenizer /path/to/telugu_bpe_6000.model \
    --reasoning-dir /content/LMA_IP_DATA/reasoning \
    --pretrain-data-dir /content/LMA_IP_DATA/train \
    --output-base /content/drive/MyDrive/LMA_IP/Telugu/finetune/mixed_data_size \
    --reasoning-sizes 2000 5000 8000 12000 16000 20000 \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --seed 42
```

The resulting directories are named according to the mixture:

```text
reasoning_2k_real_18k/
reasoning_5k_real_15k/
reasoning_8k_real_12k/
reasoning_12k_real_8k/
reasoning_16k_real_4k/
reasoning_20k_real_0k/
```

Each run contains its own generated mixed training data, validation data, configuration/metadata, training output, and checkpoints.

## 13.5 Standard single-data-size pipeline

The standard fine-tuning implementation is also provided:

```text
finetune/finetune.py
finetune/finetune_colab.py
```

It remains runnable independently, but **it is not the pipeline used for the submitted Phase 3 Telugu results**.

Example:

```bash
python -m Telugu.finetune.finetune_colab \
    --model-path /path/to/pretrained_final.pt \
    --tokenizer /path/to/telugu_bpe_6000.model \
    --data-dir /content/LMA_IP_DATA/reasoning \
    --output-dir /content/drive/MyDrive/LMA_IP/Telugu/finetune \
    --max-steps 2000 \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --seed 42
```

Resume:

```bash
python -m Telugu.finetune.finetune_colab \
    --resume /path/to/phase3_step_XXXXXX.pt \
    --tokenizer /path/to/telugu_bpe_6000.model \
    --data-dir /content/LMA_IP_DATA/reasoning \
    --output-dir /content/drive/MyDrive/LMA_IP/Telugu/finetune \
    --seed 42
```

---

# 14. Phase 3 Evaluation

The submitted Phase 3 experiment has separate evaluation utilities for the mixed-data runs.

## 14.1 Reasoning evaluation

Standard reasoning evaluation:

```text
eval/finetune/reasoning_eval.py
```

Mixed-data evaluation:

```text
eval/finetune/reasoning_eval_mixed.py
```

The evaluation compares the pretrained checkpoint and each fine-tuned checkpoint on the held-out reasoning test set.

The outputs include:

- exact-match/accuracy metrics;
- task-level metrics;
- difficulty-level metrics;
- attribute-level metrics;
- entity-type metrics;
- relation confusion;
- prediction files;
- qualitative successes;
- qualitative failures;
- answer-length statistics;
- paired comparisons.

## 14.2 Mixed-data reasoning evaluation

Use:

```bash
python -m Telugu.eval.finetune.reasoning_eval_mixed \
    --help
```

to view the exact command-line arguments for the installed version.

The mixed-data evaluation should be run separately for:

```text
reasoning_2k_real_18k
reasoning_5k_real_15k
reasoning_8k_real_12k
reasoning_12k_real_8k
reasoning_16k_real_4k
reasoning_20k_real_0k
```

The corresponding output directories are under:

```text
Telugu/results/finetune/
```

## 14.3 Training dynamics

Training-history and curve generation is implemented in:

```text
eval/finetune/training_dynamics.py
```

It can generate:

```text
training loss curves
validation loss curves
perplexity curves
learning-rate curves
training histories
training summaries
```

The existing mixed-data result directories follow the pattern:

```text
results/finetune/reasoning_<reasoning_size>k_real_<real_size>k/
├── training/
├── reasoning/
└── attention/
```

## 14.4 Pretrained vs. fine-tuned attention

Standard comparison:

```text
eval/finetune/attention_compare.py
```

Mixed-data comparison:

```text
eval/finetune/attention_compare_mixed.py
```

The mixed-data attention comparison is run for the submitted experiment's individual mixture directories.

Example:

```bash
python -m Telugu.eval.finetune.attention_compare_mixed \
    --help
```

The generated attention outputs include:

```text
attention_pretrained_early_layer.png
attention_pretrained_late_layer.png
attention_finetuned_early_layer.png
attention_finetuned_late_layer.png
attention_difference_early_layer.png
attention_difference_late_layer.png
attention_entropy_comparison.png
attention_distance_comparison.png
attention_comparison.json
attention_head_metrics.json
representative_prompt.json
```

---

# 15. Results and Logs

The repository contains generated evaluation artifacts under:

```text
results/
```

The main Phase 2 outputs include:

```text
results/
├── attention/
├── generation/
├── generations.jsonl
├── generation_samples.csv
├── generation_metrics.json
├── lm_metrics.json
├── loss_history.csv
├── learning_rate_schedule.png
├── train_vs_validation_loss.png
└── perplexity.png
```

Phase 3 outputs are organized under:

```text
results/finetune/
```

with separate directories for each mixed-data-size experiment.

Training logs are stored in:

```text
training_output.txt
```

The Bonus has its own separate results and training log under:

```text
Bonus/results/
Bonus/training_output.txt
```

The README describes how these outputs are generated; their interpretation is kept in the report.

---

# 16. Testing

Tests are located in:

```text
tests/
```

Run the complete test suite from the project root:

```bash
pytest Telugu/tests
```

The tests cover:

```text
model construction
model configuration
causal masking
forward pass
loss/backpropagation
tiny-data overfitting
real-data loading
full dataset behavior
validation data
checkpoint save/load
training smoke tests
```

Run the core model and masking tests:

```bash
pytest Telugu/tests/test_model.py Telugu/tests/test_casual_mask.py
```

Run checkpoint tests:

```bash
pytest Telugu/tests/test_checkpoint.py
```

Run the training smoke test:

```bash
pytest Telugu/tests/test_training_smoke.py
```

Tests that require the large dataset or checkpoint artifacts should be run after those artifacts have been downloaded.

---

# 17. Important Path Conventions

For Colab, the intended project location is:

```text
/content/drive/MyDrive/LMA_IP/
└── Telugu/
```

Temporary local training data:

```text
/content/LMA_IP_DATA/
├── train/
├── validation/
└── reasoning/
```

Persistent pretrained checkpoints:

```text
/content/drive/MyDrive/LMA_IP/Telugu/checkpoints/
```

Persistent Phase 3 mixed-data outputs:

```text
/content/drive/MyDrive/LMA_IP/Telugu/finetune/mixed_data_size/
```

A normal local checkout can use local paths instead. The main scripts accept explicit paths where required.

---

# 18. Reproducibility Controls

The main training and fine-tuning pipelines use seed `42`.

The code controls the relevant random sources, including:

```text
Python random
NumPy
PyTorch
CUDA RNGs when CUDA is available
```

Pretraining checkpoints store the training state needed for checkpoint recovery.

The reasoning-data generator accepts a fixed seed and produces deterministic data for a given configuration.

For comparable reruns, keep the following unchanged:

```text
Python/PyTorch environment
model_config.py
train_config.py
SentencePiece tokenizer files
tokenized corpus
train/validation/test split
random seed
batch size
sequence length
training steps
optimizer settings
learning-rate schedule
checkpoint version
```

For the Phase 3 mixed-data experiment, also keep unchanged:

```text
reasoning train/validation/test files
Phase 2 real training corpus
reasoning-data sizes
real-data sampling configuration
batch size
learning rate
seed
```

The mixed-data launcher derives each run's sampling seed from the base seed and reasoning-data size so that the individual mixtures are reproducible.

GPU/CUDA/PyTorch differences can still produce small numerical differences across environments.

---

# 19. Reproducibility Checklist

To reproduce the Telugu pipeline from the supplied artifacts:

### Step 1 — Environment

```bash
pip install torch numpy sentencepiece tqdm matplotlib sacrebleu pandas
```

Use a CUDA-enabled PyTorch installation for full GPU training.

### Step 2 — Obtain artifacts

Download the required files from the Drive placeholders in Section 2:

```text
TELUGU DATASET
TELUGU TOKENIZED CORPUS
TELUGU TOKENIZER
PHASE 2 CHECKPOINTS
PHASE 3 MIXED-DATA CHECKPOINTS
REASONING DATA
```

### Step 3 — Tokenizer

If reproducing the tokenizer:

1. Obtain the prepared deduplicated Telugu corpus.
2. Open `DC_Tokenization.ipynb`.
3. Create `telugu_500M_tokenizer_training.txt`.
4. Train the candidate SentencePiece BPE tokenizers.
5. Use `telugu_bpe_6000.model` and `telugu_bpe_6000.vocab` for the final model.
6. Run the tokenizer-statistics cells if the statistics need to be regenerated.

### Step 4 — Tokenize

```bash
python -m Telugu.data.tokenize_corpus \
    --input /path/to/telugu_500M_tokenizer_training.txt \
    --tokenizer /path/to/telugu_bpe_6000.model \
    --output Telugu/data/tokenized
```

### Step 5 — Prepare splits

```bash
python -m Telugu.data.prepare_validation \
    --input Telugu/data/tokenized \
    --train-tokens 500000000 \
    --validation-tokens 10000000 \
    --test-tokens 10000000 \
    --shard-size 10000000
```

Use `prepare_test.py` as required for the final test-data preparation.

### Step 6 — Pretrain

For GPU/Colab:

```bash
python -m Telugu.train_colab
```

For local training:

```bash
python -m Telugu.train
```

Use `--resume` with an intermediate checkpoint if the run is interrupted.

### Step 7 — Phase 2 evaluation

Run the required evaluation scripts using the exact supplied checkpoint, tokenizer, and held-out data:

```bash
python -m Telugu.eval.evaluate_lm ...
python -m Telugu.eval.generate ...
python -m Telugu.eval.metrics ...
python -m Telugu.eval.attention ...
```

### Step 8 — Generate reasoning data

```bash
python -m Telugu.reasoning.generate_reasoning_data_v2 \
    --output-dir Telugu/reasoning/data \
    --seed 42
```

### Step 9 — Phase 3 mixed-data fine-tuning

The submitted Phase 3 pipeline is:

```text
finetune/finetune_colab_mixed_data_size.py
```

Start from the Phase 2 pretrained Telugu checkpoint and use the fixed Telugu SentencePiece tokenizer.

Run:

```bash
python -m Telugu.finetune.finetune_colab_mixed_data_size \
    --model-path /path/to/pretrained_final.pt \
    --tokenizer /path/to/telugu_bpe_6000.model \
    --reasoning-dir /content/LMA_IP_DATA/reasoning \
    --pretrain-data-dir /content/LMA_IP_DATA/train \
    --output-base /content/drive/MyDrive/LMA_IP/Telugu/finetune/mixed_data_size \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --seed 42
```

This produces:

```text
reasoning_2k_real_18k
reasoning_5k_real_15k
reasoning_8k_real_12k
reasoning_12k_real_8k
reasoning_16k_real_4k
reasoning_20k_real_0k
```

with one complete pass over 20,000 examples per run.

### Step 10 — Phase 3 evaluation

Use the mixed-data evaluation utilities:

```text
eval/finetune/reasoning_eval_mixed.py
eval/finetune/pretrain_eval_mixed.py
eval/finetune/attention_compare_mixed.py
eval/finetune/training_dynamics.py
```

Run:

```bash
python -m Telugu.eval.finetune.reasoning_eval_mixed --help
python -m Telugu.eval.finetune.pretrain_eval_mixed --help
python -m Telugu.eval.finetune.attention_compare_mixed --help
python -m Telugu.eval.finetune.training_dynamics --help
```

and supply the exact checkpoint/data paths for the desired mixed-data run.

---

# 20. Code-to-Task Map

| Task | Main code |
|---|---|
| Tokenizer corpus preparation | `DC_Tokenization.ipynb` |
| SentencePiece BPE training | `DC_Tokenization.ipynb` |
| Tokenizer statistics | `DC_Tokenization.ipynb`, `Tokenizer_Stats/` |
| Corpus tokenization | `data/tokenize_corpus.py` |
| Validation split preparation | `data/prepare_validation.py` |
| Test split preparation | `data/prepare_test.py` |
| Training dataset | `data/dataset.py` |
| Validation dataset | `data/validation_dataset.py` |
| Test dataset | `data/test_dataset.py` |
| Model configuration | `config/model_config.py` |
| Training configuration | `config/train_config.py` |
| Language model | `model/lm.py` |
| Transformer | `model/transformer.py` |
| Transformer block | `model/transformer_block.py` |
| Attention | `model/attention.py` |
| Positional embedding | `model/positional_embedding.py` |
| Pretraining | `train.py` |
| Colab pretraining | `train_colab.py` |
| LM evaluation | `eval/evaluate_lm.py` |
| Generation | `eval/generate.py` |
| Generation metrics | `eval/metrics.py` |
| Generation sample creation | `eval/make_generation_samples.py` |
| Attention extraction | `eval/attention.py` |
| Attention summaries | `eval/plot_attention_summary.py` |
| Local/global attention plot | `eval/plot_attention_local_global.py` |
| Loss plots | `eval/plot_loss.py` |
| Learning-rate plots | `eval/plot_lr_schedule.py` |
| Perplexity plots | `eval/plot_perplexity.py`, `eval/plot_ppl.py` |
| Generation plots | `eval/plot_generation_metrics.py` |
| Reasoning data generation | `reasoning/generate_reasoning_data_v2.py` |
| Standard fine-tuning | `finetune/finetune.py` |
| Standard Colab fine-tuning | `finetune/finetune_colab.py` |
| **Submitted mixed-data fine-tuning** | `finetune/finetune_colab_mixed_data_size.py` |
| Standard reasoning evaluation | `eval/finetune/reasoning_eval.py` |
| Mixed-data reasoning evaluation | `eval/finetune/reasoning_eval_mixed.py` |
| Mixed-data pretrained evaluation | `eval/finetune/pretrain_eval_mixed.py` |
| Standard attention comparison | `eval/finetune/attention_compare.py` |
| Mixed-data attention comparison | `eval/finetune/attention_compare_mixed.py` |
| Fine-tuning dynamics | `eval/finetune/training_dynamics.py` |
| Tests | `tests/` |

---

# 21. Artifact Summary

Large artifacts are provided separately through Google Drive.

| Artifact | Location / placeholder |
|---|---|
| Cleaned/deduplicated Telugu corpus | https://drive.google.com/file/d/1Flqz6PAJKRcpvZ-KrJ8YfodC4f84TUeh/view?usp=sharing |
| Tokenized Telugu corpus | https://drive.google.com/drive/folders/1o-rFGvDB4n7fOVreHNbRyMR_EpreEiPP?usp=sharing |
| 6K SentencePiece tokenizer (`.model`, `.vocab`) | https://drive.google.com/drive/folders/1PKsO7cOgUkQpTHNoZkM29e9JM6KwkxYM?usp=sharing |
| Phase 2 pretrained checkpoints | https://drive.google.com/drive/folders/1L2WkWaXQdpNuY563F48_AXDlnKGO_nNy?usp=sharing |
| Phase 3 mixed-data checkpoints | https://drive.google.com/drive/folders/1EeQTjbE_Walg6bSMxpxZBO3DHp3iacrC?usp=sharing |
| Synthetic reasoning dataset | https://drive.google.com/drive/folders/1cT1fxocvkX6wnJXCB5gbUkGPsTUOTuYV?usp=sharing |
| Bonus checkpoints | https://drive.google.com/drive/folders/1hcBdAboKUrI0fKphMFnZQ2Rhh3mWd5VS?usp=sharing |

No dataset-collection reproduction procedure is included here. The reproduction procedure starts from the supplied prepared Telugu corpus and covers tokenizer training, corpus tokenization, splitting, model pretraining, evaluation, reasoning-data generation, and Phase 3 mixed-data fine-tuning.
