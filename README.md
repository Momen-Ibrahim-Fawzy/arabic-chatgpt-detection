# Detecting ChatGPT-Generated Arabic Text

**Generative AI authorship verification for Arabic: translated training data, four Arabic BERT detectors,
out-of-domain testing and six adversarial attacks**
CSE740 Advanced Natural Language Processing, M.Sc. Computer & Systems Engineering, Alexandria University (Fall 2023)
Momen Ibrahim Fawzy · supervised by Prof. Nagwa El-Makky

Detectors for LLM-written text are mostly built for English. Arabic has no large human-vs-ChatGPT corpus, and
the questions that matter in practice are not answered by in-domain accuracy:

1. Can a detector trained on *translated* data recognise ChatGPT text written natively in Arabic?
2. Does it still work on a different domain than the one it was trained on?
3. How easily is it fooled by cheap edits (misspellings, look-alike characters, broken word boundaries)?

## Approach

```
HC3 (English QA) --Google Translate EN->AR--> Arabic HC3 --fine-tune--> 4 detectors
                                                                          |
         in-domain test (HC3 test, 710)  <--------------------------------+
         out-of-domain set: ARCD questions answered by ChatGPT and by humans
         6 adversarial copies of the out-of-domain ChatGPT answers
```

**Data**
- **HC3** (Guo et al., 2023), English QA part: human answers vs ChatGPT answers to the same questions.
  Splits: train 68,335 / validation 17,114 / test 710. Train is 46,833 human and 21,502 ChatGPT, validation
  11,713 and 5,401.
- **Translation.** Every question and answer is translated to Arabic (`arabic_detect/translate.py`). The
  translator retries with backoff, translates each unique string once (HC3 repeats a question for each of
  its answers) and writes to a JSONL cache, so a multi-hour run can be resumed.
- **Out-of-domain set.** Questions from ARCD (Arabic reading comprehension) answered in Arabic by ChatGPT
  (1,946 answers) and by humans (1,027 answers), all collected manually. This text is native Arabic, not
  translated, and comes from a different domain.
- **Adversarial sets** (`arabic_detect/attacks.py`). Each attack is applied to every ChatGPT answer of the
  out-of-domain set, so all six sets have the same size as the ChatGPT set:

  | Attack | What changes | Example |
  |---|---|---|
  | Random character replacement | letter → letter with the same base shape (dots moved) | تاريخ → باريج |
  | Homoglyph substitution | letter → identical-looking code point (ي→ی U+06CC, ك→ک U+06A9, ة→ۃ ...) | العربية → العربیة |
  | Phonetic substitution | spelling that follows pronunciation (ذ/ز, ث/س, ظ/ض, ة/ه, hamza forms) | الصباح → السباح |
  | Keyboard layout errors | letter → horizontal neighbour on the Arabic keyboard | الطالب → الطللب |
  | Word segmentation errors | words joined, split inside, or the article/conjunction detached | الصباح وقرأ → الصباحوقرأ |
  | Hybrid | all five in sequence | |

  Attacks are seeded (row *i* uses `seed + i`), so every set can be regenerated exactly.

**Detectors** (`arabic_detect/models.py`, `arabic_detect/engine.py`). Each model is a pre-trained encoder
with a 2-way classification head, fine-tuned on Arabic HC3:

| Short name | Model | Pre-training |
|---|---|---|
| `camelbert-da` | CAMeL-Lab/bert-base-arabic-camelbert-da | dialectal Arabic |
| `camelbert-msa` | CAMeL-Lab/bert-base-arabic-camelbert-msa | Modern Standard Arabic |
| `camelbert-mix` | CAMeL-Lab/bert-base-arabic-camelbert-mix | MSA + dialectal + classical |
| `gigabert-v4` | lanwuwei/GigaBERT-v4-Arabic-and-English | English-Arabic, code-switched |

Shared setup: 128 tokens, batch 128, AdamW at 3e-5 with a linear schedule and 0.1% warmup, 20 epochs. The
model is evaluated every epoch, and the checkpoint with the best validation macro-F1 is kept.

## Results

**In domain: Arabic HC3 test set (710)**

| Model | Accuracy | Precision | Recall | F1 macro |
|---|:---:|:---:|:---:|:---:|
| camelbert-da | 0.984 | 0.966 | 0.984 | 0.983 |
| camelbert-msa | 0.991 | 0.980 | 0.990 | 0.990 |
| camelbert-mix | 0.990 | 0.970 | 0.987 | 0.989 |
| **GigaBERT-v4** | **0.994** | **0.984** | **0.993** | **0.995** |

**Out of domain: native Arabic ChatGPT vs human answers to ARCD questions**

| Model | Accuracy | Precision | Recall | F1 macro |
|---|:---:|:---:|:---:|:---:|
| camelbert-da | 0.548 | 0.956 | 0.309 | 0.538 |
| camelbert-msa | 0.658 | 0.975 | 0.477 | 0.657 |
| camelbert-mix | 0.621 | 0.964 | 0.425 | 0.619 |
| **GigaBERT-v4** | **0.696** | **0.981** | **0.543** | **0.696** |

**Adversarial: share of attacked ChatGPT answers still detected as ChatGPT**

| Attack | camelbert-da | camelbert-msa | camelbert-mix | GigaBERT-v4 |
|---|:---:|:---:|:---:|:---:|
| none (original ChatGPT answers) | 0.639 | 0.658 | 0.621 | 0.649 |
| random character replacement | 0.428 | 0.451 | 0.404 | **0.462** |
| homoglyph substitution | 0.451 | 0.477 | 0.425 | **0.481** |
| phonetic substitution | 0.451 | 0.477 | 0.425 | **0.486** |
| keyboard layout errors | 0.431 | 0.457 | 0.405 | **0.464** |
| word segmentation errors | 0.044 | 0.033 | 0.055 | **0.064** |
| hybrid | 0.068 | 0.065 | 0.072 | **0.086** |

What the numbers say:
- **Translated training data is enough in domain.** All four detectors score above 0.98 macro-F1 on
  translated HC3. GigaBERT-v4's bilingual pre-training gives it the edge.
- **Generalisation is the real problem.** On native Arabic answers from another domain, accuracy falls to
  0.55–0.70. Precision stays high (0.96–0.98), but recall collapses: the detectors miss ChatGPT text much
  more often than they accuse humans. Part of what they learned is the style of *translated* ChatGPT text.
- **Character noise costs about 20 points** of detection rate. **Broken word boundaries defeat the
  detectors completely** (≤ 6.4% still detected), and the hybrid attack is just as effective. Sub-word
  tokenisation is fragile when words are merged or split.
- GigaBERT-v4 is the most robust model under every attack. Among the CAMeLBERT variants, MSA pre-training
  generalises best.

Train and validation tables and notes on reading the adversarial numbers are in [`docs/results.md`](docs/results.md).

## Repository layout

```
src/arabic_detect/
  data.py        HC3 download, split tables (.xlsx/.csv/.jsonl), label normalisation
  translate.py   EN->AR translation with retries, de-duplication and a resumable JSONL cache
  attacks.py     the six Arabic adversarial attacks (seeded, rate-controlled)
  models.py      the four encoders
  engine.py      Hugging Face Trainer fine-tuning (best macro-F1 checkpoint) and a batched Detector
  metrics.py     accuracy, macro precision/recall/F1, ROC-AUC, confusion matrix (NumPy only)
  logs.py        Trainer log -> per-epoch table and training curves
scripts/  translate_hc3.py · make_attack_sets.py · train.py · evaluate.py · plot_training_log.py
configs/train.yaml
tests/    unit tests for attacks, metrics, data I/O, translation cache and log parsing
```

## Usage

```bash
pip install -e ".[train,translate,plot,dev]"

# 1. Arabic HC3
python scripts/translate_hc3.py --out data/hc3_ar --cache data/cache/en_ar.jsonl

# 2. train the four detectors
for m in camelbert-da camelbert-msa camelbert-mix gigabert-v4; do
  python scripts/train.py --config configs/train.yaml --model $m
done

# 3. adversarial sets from the out-of-domain file (columns: arabic_answer, label)
python scripts/make_attack_sets.py --source data/ood/generated.xlsx --out data/attacks --rate 0.1

# 4. evaluate everything
python scripts/evaluate.py --checkpoints runs/*/best \
  --sets test=data/hc3_ar/test.xlsx ood=data/ood/generated.xlsx \
         random_char=data/attacks/random_char.xlsx homoglyph=data/attacks/homoglyph.xlsx \
         phonetic=data/attacks/phonetic.xlsx keyboard=data/attacks/keyboard.xlsx \
         word_segmentation=data/attacks/word_segmentation.xlsx hybrid=data/attacks/hybrid.xlsx

python -m pytest tests
```

Datasets are not redistributed: HC3 is downloaded from the Hugging Face hub, and the out-of-domain answers
stay outside the repository.

## References
- B. Guo et al. *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection.* arXiv 2023.
- W. Antoun et al. *Towards a Robust Detection of Language Model Generated Text: Is ChatGPT that Easy to Detect?* arXiv 2023.
- G. Inoue et al. *The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models.* WANLP 2021.
- W. Lan et al. *GigaBERT: Zero-shot Transfer Learning from English to Arabic.* EMNLP 2020.
- H. Mozannar et al. *Neural Arabic Question Answering* (ARCD). WANLP 2019.
- M. Wolff, S. Wolff. *Attacking Neural Text Detectors.* arXiv 2020.

Code in this repository: MIT licence.
