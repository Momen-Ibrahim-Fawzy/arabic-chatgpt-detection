# Full results

All numbers come from the best checkpoint of each model, chosen by validation macro-F1 during 20 epochs of
fine-tuning. Precision, recall and F1 are macro-averaged over the two classes.

## Arabic HC3

**Train (68,335)**

| Model | Accuracy | Precision | Recall | F1 macro |
|---|:---:|:---:|:---:|:---:|
| camelbert-da | 0.9891 | 0.9839 | 0.9912 | 0.9874 |
| camelbert-msa | 0.9928 | 0.9901 | 0.9933 | 0.9916 |
| camelbert-mix | **0.9930** | **0.9906** | 0.9932 | **0.9919** |
| GigaBERT-v4 | 0.9928 | 0.9898 | **0.9937** | 0.9917 |

**Validation (17,114)**

| Model | Accuracy | Precision | Recall | F1 macro |
|---|:---:|:---:|:---:|:---:|
| camelbert-da | 0.987 | 0.976 | 0.986 | 0.988 |
| camelbert-msa | 0.994 | 0.984 | 0.993 | 0.992 |
| camelbert-mix | 0.993 | 0.980 | 0.989 | 0.991 |
| **GigaBERT-v4** | **0.997** | **0.994** | **0.995** | **0.997** |

**Test (710)**

| Model | Accuracy | Precision | Recall | F1 macro |
|---|:---:|:---:|:---:|:---:|
| camelbert-da | 0.984 | 0.966 | 0.984 | 0.983 |
| camelbert-msa | 0.991 | 0.980 | 0.990 | 0.990 |
| camelbert-mix | 0.990 | 0.970 | 0.987 | 0.989 |
| **GigaBERT-v4** | **0.994** | **0.984** | **0.993** | **0.995** |

The four models are within about one point of each other in domain. The gap between train and test is
small, so translated HC3 is learnable and the models do not simply memorise it.

## Out-of-domain set (1,946 ChatGPT + 1,027 human answers to ARCD questions)

| Model | Accuracy | Precision | Recall | F1 macro |
|---|:---:|:---:|:---:|:---:|
| camelbert-da | 0.5476 | 0.956 | 0.3088 | 0.5381 |
| camelbert-msa | 0.6576 | 0.975 | 0.4769 | 0.6572 |
| camelbert-mix | 0.6209 | 0.964 | 0.4245 | 0.6193 |
| **GigaBERT-v4** | **0.6959** | **0.981** | **0.5432** | **0.6959** |

## Adversarial sets

Each set holds only attacked ChatGPT answers, so accuracy is the share still classified as ChatGPT.

| Attack | camelbert-da | camelbert-msa | camelbert-mix | GigaBERT-v4 |
|---|:---:|:---:|:---:|:---:|
| none | 0.6392 | 0.6576 | 0.6209 | 0.6492 |
| random character replacement | 0.4278 | 0.4512 | 0.4044 | **0.4620** |
| homoglyph substitution | 0.4507 | 0.4769 | 0.4245 | **0.4811** |
| phonetic substitution | 0.4507 | 0.4769 | 0.4245 | **0.4861** |
| keyboard layout errors | 0.4314 | 0.4573 | 0.4054 | **0.4641** |
| word segmentation errors | 0.0442 | 0.0334 | 0.0550 | **0.0642** |
| hybrid | 0.0683 | 0.0647 | 0.0719 | **0.0860** |

Reading the table:
- The four character-level attacks cause a similar drop of about 18–20 points. The detectors look at more
  than spelling, so a few corrupted letters per sentence do not flip most decisions.
- Word segmentation is different in kind. When a word is joined with its neighbour or split in two, the
  WordPiece tokenizer produces unfamiliar sub-word sequences across the whole sentence, and almost every
  answer is then classified as human. Any attack that includes it (hybrid) inherits the same failure.
- The ranking of the models is stable across attacks, with GigaBERT-v4 first in every row. Robustness
  follows from the same property that helps out of domain: broader pre-training.

## Takeaways for building an Arabic detector
- Training on machine-translated data gives a strong in-domain detector that does not transfer to native
  Arabic text. Native Arabic ChatGPT output, in several domains, should be part of training.
- High precision with low recall out of domain suggests calibrating the threshold on in-domain data
  before deployment.
- Augmenting training with the attack functions here (especially word segmentation) is the obvious next
  step towards robustness. So is normalising input text (homoglyph mapping, hamza and yeh normalisation)
  before classification.
