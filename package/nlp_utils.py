import spacy
import pandas as pd


def _extract_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for k in ("title", "description", "text"):
            if k in value and value[k]:
                parts.append(value[k])
        return " ".join(parts)
    try:
        parts = [_extract_text(item) for item in value]
        return " ".join(p for p in parts if p)
    except TypeError:
        return str(value)


def add_ner_columns(df: pd.DataFrame, field: str, model: str = "en_core_web_trf", use_gpu: bool = False) -> pd.DataFrame:
    """Add NER entity columns for ``field``."""
    if use_gpu:
        if not spacy.require_gpu():
            raise RuntimeError("spaCy GPU support is not available.")
        nlp = spacy.load(model)
    else:
        nlp = spacy.load(model)

    texts = [_extract_text(v) for v in df[field]]

    entities = []
    labels = []
    for doc in nlp.pipe(texts, disable=["tagger", "parser", "lemmatizer"]):
        ents = [(e.text, e.label_) for e in doc.ents]
        entities.append([t for t, _ in ents])
        labels.append([l for _, l in ents])

    out = df.copy()
    out[f"{field}_entities"] = entities
    out[f"{field}_entity_types"] = labels
    return out


def add_nounchunk_columns(df: pd.DataFrame, field: str, model: str = "en_core_web_sm", use_gpu: bool = False) -> pd.DataFrame:
    """Add noun chunk columns for ``field``."""
    if use_gpu:
        if not spacy.require_gpu():
            raise RuntimeError("spaCy GPU support is not available.")
        nlp = spacy.load(model)
    else:
        nlp = spacy.load(model)

    texts = [_extract_text(v) for v in df[field]]

    noun_chunks = []
    for doc in nlp.pipe(texts, disable=["tagger", "parser", "lemmatizer"]):
        noun_chunks.append([chunk.text for chunk in doc.noun_chunks])

    out = df.copy()
    out[f"{field}_noun_chunks"] = noun_chunks
    return out
