# Project timeline

The project ran over the Fall 2023 term in three graded stages.

## 1. Proposal
- **Question:** can human-written and ChatGPT-written *Arabic* text be told apart reliably, given that no
  Arabic detection corpus existed?
- **Reference work:** Guo et al. (2023), who built HC3 and English/Chinese detectors.
- **Plan:**
  - an Arabic training set built by machine-translating an English human-vs-ChatGPT corpus (HC3; the LLM
    imitation dataset of Hayawi et al. was the alternative)
  - a native Arabic set, made by prompting ChatGPT with questions from Arabic QA datasets in SQuAD format
  - transformer detectors fine-tuned on both

## 2. Survey
- **In-scope work:**
  - HC3 and its detectors (Guo et al.)
  - Antoun et al. (2023) on robust detection and how easily ChatGPT detectors are attacked
  - feature-based and perplexity-based detectors (e.g. Ma et al., Desaire et al.)
- **Wider scope:** deepfake-text detection, essay-level studies (Herbold et al.) and detector-testing
  frameworks (Weber-Wulff et al.).
- **Effect on the project:** the survey added the robustness question, so the final project includes
  out-of-domain testing and adversarial attacks.

## 3. Final report
- **Arabic HC3:** HC3 translated EN→AR and used to fine-tune four Arabic encoders (CAMeLBERT-DA/MSA/Mix
  and GigaBERT-v4).
- **Out-of-domain set:** ARCD questions answered by ChatGPT (1,946) and by humans (1,027).
- **Attacks:** six adversarial copies of the ChatGPT answers.
- **Findings** (see the [README](../README.md)):
  - above 0.98 macro-F1 in domain
  - 0.55–0.70 accuracy out of domain
  - near-total failure under word-segmentation attacks
