# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package for processing and analyzing text data, particularly focused on community/social media content. The codebase has been refactored from Jupyter notebooks into a modular package for programmatic use.

## Development Workflow

Install dependencies:
```bash
pip install -r requirements.txt
```

No formal test suite exists - verify functionality using `example_pipeline.ipynb` which demonstrates the complete workflow.

## Architecture

The package provides a three-stage text processing pipeline:

1. **Data Extraction & Parsing** (`parsing.py`): Discord message parsing, URL extraction, RSS feeds, Google Sheets integration, and web scraping
2. **NLP Feature Extraction** (`nlp_utils.py`): SpaCy-based named entity recognition and noun chunk extraction
3. **Vectorization & Analysis** (`mapping.py`): Text chunking, transformer embeddings, dimensionality reduction (PaCMAP/UMAP), clustering (HDBSCAN), and topic modeling (BERTopic)

### Core Data Flow

The typical pipeline follows this pattern:
```python
# 1. Parse messages and extract content
parsed_messages = [parse_message(msg) for msg in raw_messages]
df = create_combined_content(pd.DataFrame(parsed_messages))

# 2. Add NLP features  
df = add_ner_columns(df, 'combined_content')
df = add_nounchunk_columns(df, 'combined_content')

# 3. Generate embeddings and cluster
chunks = chunk(df['combined_content'].tolist())
vectors = make_embedding(chunks)
reduced_vecs, map_vecs = reduce_vectors(vectors)
labels = cluster(reduced_vecs)
```

### Key Implementation Details

- **GPU Support**: Both NLP processing (`use_gpu=True`) and embedding generation utilize GPU acceleration when available
- **Batch Processing**: Large datasets are processed in batches with progress bars via `batch_generator()`
- **Multi-format Input**: Functions in `mapping.py` handle CSV, PDF, and zip file inputs through `load_documents()` and `unzip_files()`
- **Concurrent Processing**: URL extraction and web scraping use ThreadPoolExecutor for parallel requests

## Package Structure

All functions are exported through `__init__.py` with a comprehensive `__all__` list. The three modules are:

- **parsing.py**: 17 exported functions for content extraction and preprocessing
- **nlp_utils.py**: 2 exported functions for SpaCy-based NLP processing  
- **mapping.py**: 24 exported functions for vectorization, clustering, and topic modeling

## Important Notes

- All text processing expects pandas DataFrame input format
- SpaCy models default to `en_core_web_trf` but can be configured
- Embedding models default to transformer-based models via the `transformers` library
- No configuration files or build scripts - package is imported directly after pip install