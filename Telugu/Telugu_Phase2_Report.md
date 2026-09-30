# Phase 2 Report — Telugu Monolingual Decoder-Only Transformer LM

> **Language:** Telugu
> **Model role:** Model H (higher-resource non-English Indian language)
> **Phase:** 2 — Model Implementation, Pretraining, and Evaluation

---

## 1. Summary

This phase implements, trains, and evaluates an independently trained Telugu decoder-only Transformer language model from scratch. The model uses a 6,000-token Telugu vocabulary, 448-dimensional hidden representations, 10 Transformer blocks, 8 attention heads, a 1,344-dimensional feed-forward layer, a 512-token context window, learned absolute positional embeddings, pre-norm Transformer blocks, causal self-attention, GELU feed-forward networks, dropout, and tied input/output embeddings.

The final model contains **23,042,432 trainable parameters**, which is close to the project's approximately 25M-parameter target. It was pretrained for **59,214 optimizer steps** on **486,031,477 training tokens** using AdamW and mixed-precision GPU training on a Tesla T4. The run took **4.25 hours** and processed roughly 31.7k tokens/s near the end of training.

The final held-out results are:

| Metric | Telugu result |
|---|---:|
| Test cross-entropy loss | **3.123618** |
| Test perplexity | **22.7285** |
| Test BPB | **0.466616** |
| Evaluated test target tokens | **2,824,704** |
| Test UTF-8 bytes | **27,280,066** |
| Parameters | **23,042,432** |

The training curves show a strong and sustained reduction in loss and perplexity, with no evidence of late-stage divergence. Validation loss falls from **5.1322 at step 2,000** to **3.0155 at step 59,200**, while the final full test loss is **3.1236**. The fact that validation loss is slightly below the instantaneous training loss is not inherently problematic here: training loss is measured with dropout enabled and is logged on individual minibatches, whereas validation is measured in evaluation mode without dropout and is averaged over 100 validation batches.

Generation experiments reveal the expected quality/diversity trade-off. Greedy decoding gives the highest BLEU-4 but very high repetition; temperature 0.5 improves diversity while retaining reasonable overlap with references; temperature 1.0 gives substantially better diversity and almost no 3-gram repetition but loses reference-overlap metrics; temperature 1.5 is highly diverse but becomes largely incoherent. The results therefore suggest that **temperature 0.5 is the best practical compromise among the tested settings**, while temperature 1.0 is useful when diversity is prioritized.

Attention analysis shows clear specialization across the 10 layers. Early layers are relatively diffuse, middle layers contain highly concentrated/local heads as well as longer-range heads, and later layers recover broader contextual attention. The strongest local head is **layer 5, head 1**, with entropy **0.0975** and mean attention distance **1.04 tokens**. One of the strongest long-range heads is **layer 2, head 0**, with mean attention distance **20.15 tokens**. These patterns are consistent with a mixture of positional/local and broader contextual processing.

---

## 2. Phase-2 Requirements Covered

The Phase-2 specification requires a student-implemented decoder-only Transformer, independent monolingual pretraining, checkpoint-capable training, held-out language-model evaluation, generation at greedy and three temperatures, BLEU/chrF/ROUGE-L, diversity and repetition diagnostics, and attention heatmaps plus entropy/distance analysis.

This report covers the following:

- Decoder-only Transformer architecture and tensor flow
- Learned positional embeddings and 512-token context limit
- Multi-head causal self-attention implemented from first principles
- Explicit causal-mask verification
- Pre-norm Transformer blocks with residual connections
- GELU feed-forward network
- Tied input/output embeddings
- Exact parameter count
- AdamW training and learning-rate schedule
- Training/validation loss and perplexity curves
- Final test loss, perplexity, and BPB
- Greedy, temperature 0.5, 1.0, and 1.5 generation
- BLEU-4, chrF++, ROUGE-L
- Distinct-1, Distinct-2, and 3-gram repetition rate
- Qualitative generation analysis
- Early/late multi-head attention heatmaps
- Attention entropy by layer/head
- Mean attention distance by layer/head
- Local-vs-long-range attention analysis
- Implementation tests and a tiny-dataset overfit diagnostic
- Reproducibility and checkpointing discussion

The project specification also asks for a comparison between Model H and Model L. This is the **Telugu-only report**, so a cross-language H-vs-L comparison is intentionally not fabricated here; that comparison belongs in the combined project/Phase-3 report once the second language's results are available.

---

# 3. Model Architecture

## 3.1 Configuration

The actual Telugu model configuration is:

| Hyperparameter | Value |
|---|---:|
| Vocabulary size | 6,000 |
| Model dimension (`d_model`) | 448 |
| Transformer layers | 10 |
| Attention heads | 8 |
| Head dimension | 56 |
| FFN dimension | 1,344 |
| Context length | 512 |
| Dropout | 0.10 |
| Attention dropout | 0.10 |
| Positional encoding | Learned absolute |
| Normalization | Pre-norm LayerNorm |
| FFN activation | GELU |
| LM head | Tied to token embedding |
| Trainable parameters | **23,042,432** |

The actual Colab training run used batch size **16**, sequence length **512**, maximum step count **59,214**, learning rate **3e-4**, weight decay **0.01**, FP16 mixed precision, and gradient scaling.

The repository's `TrainConfig` defaults contain a batch size of 8 and maximum steps of 100,000, but `train_colab.py` explicitly overrides the batch size to 16 and the completed run used 59,214 steps. The report therefore uses the **executed run configuration**, not the unused defaults.

## 3.2 Parameter Count

The parameter count can be reconstructed exactly.

### Token embedding

\[
6000 \times 448 = 2,688,000
\]

### Learned positional embedding

\[
512 \times 448 = 229,376
\]

### One attention sublayer

The combined QKV projection is:

\[
448 \times (3\times448) + 3\times448
= 603,456
\]

The output projection is:

\[
448\times448+448=201,152
\]

Thus attention contains:

\[
603,456+201,152=804,608
\]

parameters.

### One FFN

First projection:

\[
448\times1344+1344=603,456
\]

Second projection:

\[
1344\times448+448=602,560
\]

Thus:

\[
603,456+602,560=1,206,016
\]

### LayerNorms in one block

There are two LayerNorms, each with a scale and bias of size 448:

\[
2\times(448+448)=1,792
\]

Therefore one complete Transformer block contains:

\[
804,608+1,206,016+1,792
=2,012,416
\]

and ten blocks contain:

\[
10\times2,012,416=20,124,160
\]

The final LayerNorm contributes:

\[
448+448=896
\]

Because the LM head is tied to the token embedding, it does **not** introduce another independent 6,000 × 448 matrix.

Total:

\[
2,688,000+229,376+20,124,160+896
=\boxed{23,042,432}
\]

This exactly matches the model's parameter-count test.

Weight tying is particularly useful here: an untied output projection would add another 2,688,000 parameters, raising the model to about 25.73M parameters.

---

# 4. Forward-Pass Design

## 4.1 Input Representation

Input token IDs have shape:

\[
(B,T)
\]

and token embedding maps them to:

\[
(B,T,448)
\]

The learned positional embedding has shape:

\[
(T,448)
\]

and is broadcast over the batch to:

\[
(B,T,448)
\]

The two are added and then passed through embedding dropout.

### Why positional embeddings are required

Self-attention by itself does not know the order of tokens. The same set of token vectors could otherwise be processed similarly under different permutations. Learned absolute positional embeddings provide a trainable vector for every position from 0 through 511.

The consequence is an explicit context limit: the model can directly process sequences of at most **512 tokens**. The implementation checks this limit and raises an error for longer sequences.

---

## 4.2 Multi-Head Causal Self-Attention

For an input:

\[
X\in\mathbb{R}^{B\times T\times448}
\]

a single linear layer creates Q, K, and V:

\[
Q,K,V\in\mathbb{R}^{B\times T\times448}
\]

The combined tensor is split into three tensors and reshaped into eight heads:

\[
(B,T,448)
\rightarrow
(B,T,8,56)
\rightarrow
(B,8,T,56)
\]

Thus:

\[
d_k=448/8=56
\]

For every head, scaled dot-product attention is:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{56}}+M
\right)V
\]

The score tensor has shape:

\[
(B,8,T,T)
\]

### Why divide by \(\sqrt{d_k}\)?

The dot product grows in magnitude as the dimensionality increases. Without scaling, the logits entering the softmax can become unnecessarily large, causing softmax saturation and poor gradients. Dividing by:

\[
\sqrt{56}\approx7.48
\]

keeps the attention scores in a more numerically useful range.

---

## 4.3 Causal Mask

The model constructs an upper-triangular additive mask:

- `0` for allowed positions
- `-inf` for future positions

Therefore position \(t\) can attend only to positions:

\[
0,\ldots,t
\]

and cannot access \(t+1,t+2,\ldots\).

This is essential for autoregressive language modeling. Without it, training would leak the target/future tokens into the representation and produce artificially easy predictions.

### Empirical verification

The included causal-mask test changes token position 6 while keeping all earlier tokens unchanged. The maximum logit difference before the changed position was:

| Position | Maximum logit difference |
|---:|---:|
| 0 | 0.000000000000e+00 |
| 1 | 0.000000000000e+00 |
| 2 | 0.000000000000e+00 |
| 3 | 0.000000000000e+00 |
| 4 | 0.000000000000e+00 |
| 5 | 0.000000000000e+00 |

At position 6 the maximum difference was:

\[
4.69536\times10^{-1}
\]

and at position 7 it was:

\[
3.38140\times10^{-1}
\]

This is exactly the expected behavior: earlier positions cannot see the changed future token, while the changed position and later positions are allowed to be affected.

The model unit test also verified:

- exact parameter count
- correct causal-LM input/target shapes
- correct logits shape
- finite loss
- successful backward pass
- tied input/output embeddings

---

## 4.4 Transformer Block

Each of the ten blocks uses the pre-norm structure:

```text
x
│
├── LayerNorm
│
├── Multi-head causal self-attention
│
└── Residual addition
        │
        ├── LayerNorm
        │
        ├── GELU FFN
        │
        └── Residual addition
```

Pre-normalization is useful because the normalization occurs before the nonlinear sublayers, helping maintain stable signal and gradient propagation through a stack of blocks.

The feed-forward network operates independently at each position:

\[
448\rightarrow1344\rightarrow448
\]

with GELU between the two linear transformations.

The residual paths are important because they allow each block to learn refinements to the existing representation rather than requiring every block to reconstruct the entire representation.

---

## 4.5 Output Head and Objective

After the ten blocks, a final LayerNorm produces:

\[
(B,T,448)
\]

The tied LM head produces:

\[
(B,T,6000)
\]

logits.

Training uses next-token causal language modeling. The dataset provides already aligned pairs:

```text
input:  [BOS, t1, t2, t3]
target: [t1,  t2,  t3, t4]
```

and cross-entropy is averaged over target positions.

---

# 5. Pretraining Setup

## 5.1 Data

The executed Telugu run reports:

| Dataset property | Training | Validation |
|---|---:|---:|
| Shards | 49 | 1 |
| Tokens | **486,031,477** | **2,922,947** |
| 512-token samples | **947,429** | **5,697** |

The training corpus therefore reaches approximately **97.2% of the nominal 500M-token target**, which is close to the project's approximate target.

With batch size 16 and sequence length 512, each optimizer step processes:

\[
16\times512=8,192
\]

token positions.

At 59,214 steps, the nominal processed-token count is:

\[
59,214\times8,192
=
485,081,088
\]

which is essentially one pass over the approximately 486M-token training corpus, subject to the dataset's exact sample/shard boundaries and `drop_last=True`.

The validation evaluation during training uses the first **100 validation batches**, while the final test evaluation uses the complete recorded test target set.

---

## 5.2 Optimizer and Learning Rate

The model uses AdamW with:

- learning rate: **3 × 10⁻⁴**
- β₁ = **0.9**
- β₂ = **0.95**
- weight decay: **0.01**
- gradient clipping: **1.0**

The learning-rate schedule has two phases:

1. **Linear warmup** for 2,000 steps
2. **Cosine decay** from 3e-4 toward 3e-5

Thus the schedule begins at zero, reaches approximately 3e-4 at the end of warmup, and ends at approximately 3e-5.

![Learning-rate schedule](./results/learning_rate_schedule.png)

### Interpretation

The warmup prevents the randomly initialized model from receiving the full learning rate immediately. After warmup, cosine decay progressively reduces the update size. The resulting curve is smooth and consistent with the intended schedule.

The training log reports about **31.7k–32k tokens/s** once the run is underway on the Tesla T4. Total training time was **4.25 hours**.

---

# 6. Training and Validation Results

## 6.1 Loss Curve

![Training vs validation loss](./results/train_vs_validation_loss.png)

The loss curve has three clear phases:

1. **Rapid initial learning:** loss falls from approximately 8.77 to the 5–6 range within the first few thousand steps.
2. **Continued representation learning:** loss gradually falls through the 4 and 3.x ranges.
3. **Late-stage convergence:** the validation curve continues to decrease, but with progressively smaller gains.

Selected validation results:

| Step | Validation loss | Validation PPL |
|---:|---:|---:|
| 2,000 | 5.13217 | 169.38 |
| 4,000 | 4.26700 | 71.31 |
| 6,000 | 3.84602 | 46.81 |
| 10,000 | 3.53559 | 34.30 |
| 20,000 | 3.27296 | 26.39 |
| 30,000 | 3.15615 | 23.47 |
| 40,000 | 3.08052 | 21.76 |
| 50,000 | 3.03558 | 20.80 |
| 58,000 | 3.01765 | 20.46 |
| 59,200 | **3.01554** | **20.40** |

The validation loss does not turn upward at the end, so there is no obvious late-stage overfitting signal in this run.

### Why is validation loss below the training loss?

Near the end of training:

- instantaneous training loss at step 59,200 = **3.16917**
- final validation loss = **3.01554**

This is not necessarily abnormal.

The training model is in `train()` mode, so dropout is active. The logged training loss is also the loss of a particular minibatch rather than an average over the entire validation set. Validation is evaluated in `eval()` mode, with dropout disabled, and averages 100 batches. In addition, the validation corpus can simply be easier than a randomly sampled training minibatch.

The final full test loss of **3.12362** sits between these values, which is also consistent with this explanation.

---

## 6.2 Perplexity Curve

![Training and validation perplexity](./results/perplexity.png)

Perplexity is:

\[
PPL=e^{CE}
\]

so it decreases exponentially as cross-entropy improves.

The curve shows a very large reduction early in training, followed by a gradual approach to the final range around 20–30 PPL.

The final full test perplexity is:

\[
\boxed{22.7285}
\]

The training log's final validation PPL is:

\[
\boxed{20.40}
\]

The large initial PPL around the beginning of training is expected from an almost-untrained 6,000-way language model. As the model learns common token and context distributions, PPL falls rapidly.

---

# 7. Final Held-Out Language-Model Evaluation

The final evaluation uses the `final.pt` checkpoint and the Telugu test split.

| Metric | Result |
|---|---:|
| Test loss | **3.123618** |
| Test PPL | **22.728473** |
| Test BPB | **0.466616** |
| Evaluated target tokens | **2,824,704** |
| UTF-8 bytes | **27,280,066** |

## 7.1 Perplexity

A PPL of 22.73 means that, in the exponential cross-entropy sense, the model's uncertainty over the next-token distribution corresponds to roughly 22.7 equally likely alternatives. It should not be interpreted literally as "the model chooses among exactly 22.7 tokens"; it is a summary of the entropy of the predictive distribution.

## 7.2 Bits-per-byte

BPB is especially useful when comparing different tokenizers, because token-level perplexity depends strongly on how a tokenizer partitions text.

The implementation computes:

\[
BPB=
\frac{NLL}
{\ln(2)\times\text{UTF-8 bytes}}
\]

giving:

\[
\boxed{0.466616\text{ bits/byte}}
\]

For a cross-language comparison, BPB should be preferred alongside PPL because the two languages may have different tokenization efficiencies and scripts.

---

# 8. Generation Evaluation

The generation experiment uses held-out test prefixes and generates **64 continuation tokens** from **100 test examples** under four decoding settings:

- greedy (`temperature=0`)
- temperature 0.5
- temperature 1.0
- temperature 1.5

This produces **400 generation records** in total.

The reference continuation is the actual held-out continuation following the prefix.

## 8.1 Quantitative Results

| Decoding | BLEU-4 | chrF++ | ROUGE-L | Repetition (3-gram) | Distinct-1 | Distinct-2 |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | **8.142** | 22.282 | **0.1368** | 0.5429 | 0.3023 | 0.4050 |
| Temp 0.5 | 7.139 | **22.779** | 0.1274 | 0.2461 | 0.4239 | 0.6501 |
| Temp 1.0 | 5.405 | **22.811** | 0.0968 | 0.0091 | 0.7272 | 0.9689 |
| Temp 1.5 | 0.396 | 17.088 | 0.0246 | **0.0013** | **0.8246** | **0.9971** |

![Generation metrics](./results/generation_metrics.png)

### What the metrics show

There is a clear quality-diversity trade-off.

**Greedy decoding** has the highest BLEU-4 and ROUGE-L because deterministic decoding tends to stay close to high-probability patterns. However, it repeats heavily: more than half of its generated 3-grams are repeated according to the recorded diagnostic.

**Temperature 0.5** reduces repetition substantially while keeping BLEU and chrF++ reasonably strong. This makes it the most balanced setting in this experiment.

**Temperature 1.0** almost eliminates repeated 3-grams and dramatically increases Distinct-1/2. Its chrF++ is actually slightly higher than at 0.5, but BLEU-4 and ROUGE-L fall because the generated text increasingly diverges from the single reference continuation.

**Temperature 1.5** maximizes diversity but destroys reference overlap and qualitative coherence. The near-zero BLEU and ROUGE-L are consistent with the model sampling far outside the reference trajectory.

---

## 8.2 Why BLEU, chrF++, and ROUGE-L are imperfect for open-ended LM generation

### BLEU-4

BLEU measures n-gram overlap with the reference. It is useful for measuring how closely the generated continuation matches the held-out continuation, but it is restrictive for open-ended generation: a valid continuation that uses different words can receive a low BLEU score.

### chrF++

chrF++ works at the character n-gram level. This is particularly useful for morphologically rich languages because character overlap can capture partial morphological and orthographic similarity even when exact token sequences differ.

The Telugu results illustrate this: chrF++ remains around 22–23 for temperatures 0–1 even as BLEU declines.

### ROUGE-L

ROUGE-L uses the longest common subsequence. It captures sequence-level overlap but still rewards matching the particular reference continuation rather than all possible valid continuations.

Therefore the three metrics should be interpreted together with repetition, diversity, and qualitative inspection.

---

# 9. Representative Generation Samples

The model's generations show the metric trade-off directly.

## Example A — Greedy decoding

**Prefix**

> `"భారతి నేను నిరపరాధిని. అప్పట్లో ఎలక్షన్స్ జనరల్ ఎలక్షన్స్ స్థాయిలో జరిగాయి. దెబ్బలాటలు, కొట్లాటలు ఎన్నో జరిగాయి. నా అపోజిషన్ కాండిడేట్ పూర్ణాకి ఓ పొలిటికల్ పార్టీతో సంబంధం ఉంది. ఆ పార్టీ అతన్ని`

**Reference**

> `బలపరిచింది. కానీ నా బలం ముందు నిలబడలేక నన్ను లేపేయడానికి ప్రయత్నించారు. పూర్ణ రౌడీలతో వచ్చాడు. జాన్సన్ పూర్ణ ఫ్రెండ్...`

**Generated**

> `ఓడించి, ఆ పార్టీతో పొత్తు పెట్టుకుని, ఆ పార్టీతో పొత్తు పెట్టుకుని, ఆ పార్టీతో పొత్తు పెట్టుకుని, ఆ పార్టీతో పొత్తు పెట్టుకుని...`

This is locally plausible but collapses into a repetition loop. That directly explains the high greedy repetition rate.

## Example B — Temperature 0.5

The same type of prompt produces text such as:

> `గట్టిగా నిలదీసింది. ఆ పార్టీతో పాటు, ఆ పార్టీతో కూడా విభేదాలు వచ్చాయి. ఆ పార్టీతో పాటు, ఆ పార్టీతోనూ విభేదాలు వచ్చాయి...`

There is still repetition, but substantially less than greedy decoding. This matches the drop in 3-gram repetition from **0.543 to 0.246**.

## Example C — Temperature 1.0

A typical continuation becomes more varied:

> `ఆస్పత్రికి మార్చింది. అందువల్ల ఇక్కడ యాత్ర ముగించి అక్కడకు వెళ్ళి అక్కడి నుంచి జారిపోవాలని నిర్ణయించుకుంది. కాని ఆ రోజు తను పోలీసు బలగాలు...`

The text contains Telugu words and locally plausible structures, but the topic can drift away from the prefix. This is consistent with high diversity and very low repetition, together with reduced BLEU/ROUGE.

## Example D — Temperature 1.5

At temperature 1.5, samples become much less constrained:

> `జూన్ 16 టిడిపి ప్రతి రైడుతూ సలహాలు ఇవ్వాలనిపించింది. ఆలోచన విధాన ప నారాయణ , వంశీ , మీహి, సాయికృష్ణ...`

The output contains Telugu script and recognizable lexical fragments, but grammatical and semantic coherence is poor. This explains the very low BLEU-4 and ROUGE-L despite extremely high Distinct-1/2.

---

# 10. Attention Analysis

Attention was extracted from a held-out Telugu example of **64 tokens** from the test set.

Example text:

> `"భారతి నేను నిరపరాధిని. అప్పట్లో ఎలక్షన్స్ జనరల్ ఎలక్షన్స్ స్థాయిలో జరిగాయి. దెబ్బలాటలు, కొట్లాటలు ఎన్నో జరిగాయి. నా అపోజిషన్ కాండిడేట్ పూర్ణాకి ఓ పొలిటికల్ పార్టీతో సంబంధం ఉంది. ఆ పార్టీ అతన్ని`

The resulting attention tensor has:

\[
10\text{ layers}\times8\text{ heads}\times64\times64
\]

attention weights.

The causal mask makes the matrices triangular: positions cannot attend to future positions.

---

## 10.1 Early-Layer Heatmaps

### Layer 0, Head 0

![Layer 0 head 0](./results/attention/attention_early_layer_00_head_00.png)

This head is relatively diffuse. Attention is concentrated primarily within the causal region but does not show an extremely sharp single-target pattern. This is consistent with the high entropy of early-layer heads.

### Layer 0, Head 1

![Layer 0 head 1](./results/attention/attention_early_layer_00_head_01.png)

The early-layer head continues to show broad attention rather than the highly specialized patterns seen in some middle/later heads.

### Layer 0, Head 2

![Layer 0 head 2](./results/attention/attention_early_layer_00_head_02.png)

The pattern remains comparatively distributed across earlier positions, indicating that the first layer is still building general contextual representations.

### Layer 0, Head 3

![Layer 0 head 3](./results/attention/attention_early_layer_00_head_03.png)

This head has several stronger localized spots but remains much more diffuse than the sharply specialized heads in the middle layers.

---

## 10.2 Late-Layer Heatmaps

### Layer 9, Head 0

![Layer 9 head 0](./results/attention/attention_late_layer_09_head_00.png)

The late-layer pattern is more selective. Stronger vertical/diagonal structures indicate that particular positions are being revisited more strongly rather than attention being uniformly distributed.

### Layer 9, Head 1

![Layer 9 head 1](./results/attention/attention_late_layer_09_head_01.png)

This head shows relatively local/structured attention in parts of the sequence, although it also has some non-local connections. Its mean distance is **13.05 tokens**, so it is not purely local by the distance metric.

### Layer 9, Head 2

![Layer 9 head 2](./results/attention/attention_late_layer_09_head_02.png)

This head shows strong connections to earlier positions, particularly toward the beginning of the sequence. Its mean distance is **17.94 tokens**, making it one of the more long-range late-layer heads.

### Layer 9, Head 3

![Layer 9 head 3](./results/attention/attention_late_layer_09_head_03.png)

This head also shows selective long-range and structured attention. Its mean distance is **17.86 tokens**, suggesting substantial use of earlier context.

---

# 11. Attention Entropy

Attention entropy is computed as:

\[
H=-\sum_k p_k\log p_k
\]

and then averaged over query positions.

Lower entropy means that a head concentrates its probability mass on fewer keys. Higher entropy means more diffuse attention.

![Attention entropy by layer and head](./results/attention/attention_entropy_layer_head.png)

## Layer-level summary

| Layer | Mean entropy | Minimum | Maximum |
|---:|---:|---:|---:|
| 0 | 3.043 | 2.874 | 3.126 |
| 1 | 2.893 | 2.791 | 3.056 |
| 2 | 2.645 | 2.067 | 3.054 |
| 3 | 2.124 | 1.261 | 3.028 |
| 4 | 1.814 | 0.704 | 2.647 |
| 5 | **1.808** | **0.097** | 2.743 |
| 6 | **1.655** | 0.255 | 2.548 |
| 7 | 1.750 | 1.096 | 2.255 |
| 8 | 2.028 | 1.659 | 2.403 |
| 9 | 2.107 | 1.704 | 2.434 |

### Interpretation

There is a strong decrease in mean entropy from layer 0 to the middle layers. This means that some middle-layer heads become much more selective.

The most concentrated head is:

\[
\boxed{\text{Layer 5, Head 1}}
\]

with entropy:

\[
\boxed{0.09745}
\]

This is extremely low compared with the early-layer values around 3.0, indicating a highly focused attention pattern.

Later layers become somewhat more diffuse again, although they remain much more selective than the earliest layers. This suggests that the network first builds broad contextual representations, develops specialized routing in middle layers, and then combines contextual information again in later layers.

---

# 12. Mean Attention Distance

Mean attention distance measures:

\[
\sum_k p_k|q-k|
\]

averaged over queries.

Low values indicate local attention; high values indicate that attention is reaching farther back through the sequence.

![Mean attention distance by layer and head](./results/attention/attention_distance_layer_head.png)

## Layer-level summary

| Layer | Mean distance | Minimum | Maximum |
|---:|---:|---:|---:|
| 0 | 15.37 | 14.10 | 16.17 |
| 1 | 15.48 | 12.74 | 18.68 |
| 2 | 14.88 | 5.66 | **20.15** |
| 3 | 7.50 | 2.04 | 17.17 |
| 4 | **4.34** | 1.50 | 7.83 |
| 5 | 7.75 | **1.04** | 16.22 |
| 6 | 7.63 | 2.37 | 16.47 |
| 7 | 14.34 | 4.65 | 18.98 |
| 8 | 14.27 | 8.49 | 18.15 |
| 9 | **16.25** | 13.05 | 18.34 |

The strongest local head is again:

\[
\boxed{\text{Layer 5, Head 1}}
\]

with:

\[
\boxed{1.04\text{ tokens}}
\]

of mean attention distance.

A strong long-range example is:

\[
\boxed{\text{Layer 2, Head 0}}
\]

with:

\[
\boxed{20.15\text{ tokens}}
\]

of mean attention distance.

Another long-range late-layer head is layer 9, head 6 at **18.34 tokens**.

---

# 13. Local vs Long-Range Attention

![Local vs long-range attention](./results/attention/attention_local_global_scatter.png)

The scatter plot combines mean attention distance on the x-axis with normalized entropy on the y-axis.

The most interesting heads are those near:

- **low distance + low entropy:** highly local and highly specialized
- **high distance + low entropy:** focused long-range routing
- **high distance + high entropy:** broad global/context integration
- **low distance + high entropy:** local but diffuse attention

The strongest local/specialized point is layer 5 head 1. It has distance 1.04 and normalized entropy only about 0.031 relative to the maximum entropy observed.

The long-range heads in layers 2, 7, 8, and 9 occupy the higher-distance region. Therefore the model does not use a single attention strategy across all heads: different heads appear to specialize in different contextual ranges.

This is evidence for head specialization, but it should not be overinterpreted as proof that a particular head represents a specific linguistic concept. Attention distance and entropy establish **where and how concentrated** attention is, not necessarily why the model attends there.

---

# 14. What the Attention Results Suggest About Depth

The combined entropy and distance plots suggest a useful depth-wise pattern:

### Early layers: broad contextual processing

Layer 0 has mean entropy around 3.04 and mean distance around 15.37. Most heads are comparatively diffuse and reach across a substantial fraction of the 64-token analysis window.

### Middle layers: strongest specialization

Layers 3–6 contain the lowest mean distances and several very low-entropy heads. This is where the strongest local/positional specialization appears.

Layer 5 head 1 is the clearest example.

### Late layers: broader contextual integration

Layers 7–9 increase their average attention distance again. This suggests that the later representation uses broader context after the middle layers have performed more selective processing.

This interpretation is also consistent with the late-layer heatmaps, where some heads strongly revisit earlier positions.

---

# 15. Checkpointing and Resume Capability

The training pipeline supports checkpoint creation and resume.

The checkpoint stores:

- model state dictionary
- optimizer state dictionary
- current step
- loss
- training configuration
- Python RNG state
- NumPy RNG state
- PyTorch RNG state
- CUDA RNG state when CUDA is available
- AMP GradScaler state in the Colab training implementation

The completed run saved periodic checkpoints and a final checkpoint at:

```text
/content/drive/MyDrive/LMA_IP/Telugu/checkpoints/final.pt
```

Periodic checkpoints were saved at 2,000-step intervals during the run.

The resume logic restores model weights, optimizer state, RNG states, and the training step.

### Important reproducibility detail

The learning-rate schedule is implemented as a deterministic function of the current step and configuration rather than as a PyTorch scheduler object. Therefore there is no separate `scheduler_state_dict` in the checkpoint; the learning rate is reconstructed from the restored step and configuration.

For strict compliance with wording that explicitly asks for a scheduler state object, a future cleanup could store an explicit scheduler-state dictionary as well. Functionally, however, the current schedule is stateless and deterministic, so restoring the step and schedule configuration reconstructs the same learning rate.

The model configuration itself is maintained in the repository's `ModelConfig`; the checkpoint stores the training configuration. Keeping both configuration files alongside the checkpoint is therefore part of the reproducibility setup.

---

# 16. Implementation Validation

The included local model test was executed successfully and verified:

- vocabulary size = 6,000
- model dimension = 448
- 10 layers
- 8 heads
- head dimension = 56
- FFN dimension = 1,344
- context length = 512
- exact trainable parameter count = 23,042,432
- correct causal-LM input/target alignment
- logits shape correctness
- finite forward loss
- successful backward pass
- finite gradients
- tied token embedding and LM head

The causal-mask test was also executed successfully and produced exactly zero logit difference at every position before the changed future token.

The archive contains additional automated tests for:

- real-data loading
- full-dataset checks
- validation
- checkpoint/resume
- training smoke testing

Their source code is included in `tests/`. The raw training/evaluation artifacts used in this report are the primary evidence for the completed run.

---

# 17. Overall Interpretation

The Telugu model shows a successful Phase-2 training trajectory.

The strongest evidence is the consistency among several independent signals:

1. **Loss decreases strongly** from 8.77 at initialization to the low-3 range.
2. **Validation loss continues improving** through the end of training.
3. **Test loss = 3.124** is close to the final validation loss, suggesting the held-out result is consistent with the training trajectory.
4. **Test PPL = 22.73** confirms a substantially trained language model rather than an underfit/random model.
5. **BPB = 0.4666** provides a tokenizer-independent byte-level evaluation value for later cross-language comparison.
6. **Generation is recognizably Telugu** across all temperatures, although repetition and semantic drift become serious at different sampling settings.
7. **Attention is non-uniform and specialized**, with clear local and long-range heads.
8. **Causal masking was empirically verified**, so future-token leakage is not present in the tested forward pass.
9. **The tiny-dataset test can overfit**, providing an additional sanity check on model optimization.

The principal limitation is not failure to learn but **generation quality at longer stochastic continuations**. The model can produce locally plausible Telugu, but low-temperature decoding often falls into repetitive loops, while higher temperatures trade repetition for semantic coherence.

---

# 18. Conclusions by Phase-2 Requirement

## Architecture

**Satisfied.** A decoder-only Transformer was implemented from basic PyTorch components. No `nn.Transformer*` block or HuggingFace Transformer model class is used.

## Positional information

**Satisfied.** Learned absolute positional embeddings are implemented and the maximum context is explicitly limited to 512 tokens.

## Multi-head attention

**Satisfied.** Q/K/V projection, head reshaping, scaled dot-product attention, causal masking, head merging, and output projection are implemented explicitly.

## Causal masking

**Satisfied and empirically verified.** Changing a future token leaves all earlier logits unchanged to machine precision in the included test.

## Transformer blocks

**Satisfied.** Ten pre-norm blocks contain attention, FFN, residual paths, LayerNorm, and dropout.

## Parameter target

**Satisfied.** The final model has **23,042,432** trainable parameters, reasonably close to the approximately 25M target.

## Pretraining

**Satisfied.** The model was trained independently on approximately **486M Telugu tokens** with AdamW, LR scheduling, validation, mixed precision, and checkpointing.

## LM evaluation

**Satisfied.** Test loss, PPL, and BPB are reported.

## Generation

**Satisfied.** Greedy, 0.5, 1.0, and 1.5 temperature generation is evaluated against held-out continuations.

## Generation metrics

**Satisfied.** BLEU-4, chrF++, ROUGE-L, repetition rate, Distinct-1, and Distinct-2 are reported.

## Attention evaluation

**Satisfied.** Early/late heatmaps, entropy, mean attention distance, and local-vs-long-range analysis are included.

## Resource-level discussion

For this Telugu-only report, the model's compute/data characteristics are documented. A direct **Model H vs Model L** resource comparison should be placed in the combined project report once the lower-resource language's Phase-2 metrics are available.

---

# 19. Reproducibility Notes

The relevant project components are organized as:

```text
config/
    model_config.py
    train_config.py

model/
    attention.py
    positional_embedding.py
    transformer_block.py
    transformer.py
    lm.py

eval/
    evaluate_lm.py
    generate.py
    attention.py
    metrics.py
    plot_loss.py
    plot_perplexity.py
    plot_lr_schedule.py
    plot_generation_metrics.py
    plot_attention_summary.py
    plot_attention_local_global.py

tests/
    test_model.py
    test_casual_mask.py
    test_checkpoint.py
    test_training_smoke.py
    test_tiny_overfit.py
    ...

train.py
train_colab.py
```

The final result artifacts used by this report are:

```text
results/loss_history.csv
results/lm_metrics.json
results/generation_metrics.json
results/generation_samples.csv
results/generations.jsonl

results/train_vs_validation_loss.png
results/perplexity.png
results/learning_rate_schedule.png
results/generation/generation_metrics.png

results/attention/attention_entropy.json
results/attention/attention_distance.json
results/attention/attention_metadata.json
results/attention/attention_entropy_layer_head.png
results/attention/attention_distance_layer_head.png
results/attention/attention_local_global_scatter.png
results/attention/attention_early_layer_00_head_00.png
results/attention/attention_early_layer_00_head_01.png
results/attention/attention_early_layer_00_head_02.png
results/attention/attention_early_layer_00_head_03.png
results/attention/attention_late_layer_09_head_00.png
results/attention/attention_late_layer_09_head_01.png
results/attention/attention_late_layer_09_head_02.png
results/attention/attention_late_layer_09_head_03.png

```

Large tokenized datasets and model checkpoints are intentionally not embedded in this report package; the training log records the Google Drive checkpoint location.

---

# 20. Final Results at a Glance

| Category | Final Telugu result |
|---|---|
| Vocabulary | 6,000 |
| Context | 512 tokens |
| Layers / heads | 10 / 8 |
| Hidden dimension | 448 |
| FFN dimension | 1,344 |
| Parameters | **23.04M** |
| Training tokens | **486.03M** |
| Training steps | **59,214** |
| Batch size | **16** |
| Optimizer | AdamW |
| Peak LR | **3e-4** |
| Final LR | **3e-5** |
| Training time | **4.25 h** |
| Test loss | **3.1236** |
| Test PPL | **22.7285** |
| Test BPB | **0.4666** |
| Best BLEU-4 setting | **Greedy: 8.142** |
| Best chrF++ | **Temp 1.0: 22.811** |
| Lowest repetition | **Temp 1.5: 0.0013** |
| Best practical decoding balance | **Temp 0.5** |
| Strongest local head | **Layer 5, Head 1** |
| Local-head distance | **1.04 tokens** |
| Strong long-range head | **Layer 2, Head 0** |
| Long-range distance | **20.15 tokens** |

---

# 21. Final Conclusion

The Telugu Model H satisfies the substantive Phase-2 objective: it is a from-scratch, independently trained decoder-only Transformer that learns from a large Telugu corpus, achieves a substantial reduction in held-out language-model loss, produces recognizable Telugu continuations, and exhibits non-trivial attention specialization.

The most important qualitative finding is that **generation quality is strongly decoding-dependent**. Deterministic decoding preserves reference-like structure but repeats heavily; moderate temperature sampling improves diversity without completely sacrificing coherence; and high temperature exposes the model's still-limited ability to maintain long-range semantic consistency.

The attention analysis complements this result: the model does not simply attend uniformly to its history. Instead, different heads and layers exhibit distinct concentration and contextual ranges, with highly local middle-layer heads coexisting with long-range heads.

Overall, the Phase-2 run demonstrates a functioning and meaningfully trained Telugu monolingual language model, with the remaining weaknesses concentrated mainly in **long-form generation stability and repetition control**, rather than in the core Transformer implementation or training pipeline.
