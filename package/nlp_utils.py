import spacy
import pandas as pd

# Cache for loaded models to avoid reloading
_model_cache = {}


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


def _load_model(model: str):
    """Load and cache spaCy model with GPU support."""
    if model not in _model_cache:
        try:
            spacy.require_gpu()
            _model_cache[model] = spacy.load(model)
        except OSError:
            raise RuntimeError(f"spaCy model '{model}' not found. Please install it with: python -m spacy download {model}")
        except Exception as e:
            raise RuntimeError(f"spaCy GPU support is not available: {e}")
    return _model_cache[model]


def add_ner_columns(df: pd.DataFrame, field: str, model: str = "en_core_web_trf") -> pd.DataFrame:
    """Add NER entity columns for ``field``."""
    # Validate inputs
    if field not in df.columns:
        raise ValueError(f"Column '{field}' not found in dataframe")
    
    if df.empty:
        out = df.copy()
        out[f"{field}_entities"] = []
        out[f"{field}_entity_types"] = []
        return out
    
    # Load model (cached)
    nlp = _load_model(model)

    texts = [_extract_text(v) for v in df[field]]

    entities = []
    labels = []
    for doc in nlp.pipe(texts, disable=["tagger", "parser", "lemmatizer"], batch_size=50):
        ents = [(e.text, e.label_) for e in doc.ents]
        entities.append([t for t, _ in ents])
        labels.append([l for _, l in ents])

    out = df.copy()
    out[f"{field}_entities"] = entities
    out[f"{field}_entity_types"] = labels
    return out


def add_nounchunk_columns(df: pd.DataFrame, field: str, model: str = "en_core_web_trf") -> pd.DataFrame:
    """Add noun chunk columns for ``field``."""
    # Validate inputs
    if field not in df.columns:
        raise ValueError(f"Column '{field}' not found in dataframe")
    
    if df.empty:
        out = df.copy()
        out[f"{field}_noun_chunks"] = []
        return out
    
    # Load model (cached)
    nlp = _load_model(model)

    texts = [_extract_text(v) for v in df[field]]

    noun_chunks = []
    # Note: noun chunks require the parser, so we only disable tagger and lemmatizer
    for doc in nlp.pipe(texts, disable=["tagger", "lemmatizer"], batch_size=50):
        noun_chunks.append([chunk.text for chunk in doc.noun_chunks])

    out = df.copy()
    out[f"{field}_noun_chunks"] = noun_chunks
    return out
