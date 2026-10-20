"""The four Arabic encoders compared in the project."""
from __future__ import annotations

MODELS = {
    # CAMeLBERT (Inoue et al., 2021): same architecture, different pre-training variants
    "camelbert-da": "CAMeL-Lab/bert-base-arabic-camelbert-da",    # dialectal Arabic
    "camelbert-msa": "CAMeL-Lab/bert-base-arabic-camelbert-msa",  # Modern Standard Arabic
    "camelbert-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",  # MSA + DA + Classical
    # GigaBERT-v4 (Lan et al., 2020): English-Arabic bilingual, continued on code-switched data
    "gigabert-v4": "lanwuwei/GigaBERT-v4-Arabic-and-English",
}


def resolve(name: str) -> str:
    """Accept a short name from ``MODELS`` or any Hugging Face model id / local path."""
    return MODELS.get(name.lower(), name)
