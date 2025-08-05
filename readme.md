# Convenient wrappers for analyzing text data from online communities and the across the web

## Package layout

```
community_collection/
    __init__.py
    parsing.py         # message parsing helpers
    nlp_utils.py       # spaCy based NLP helpers
    mapping.py         # vectorisation, clustering and topic modelling
```

## Module summaries

**parsing.py**

- `parse_message` extracts message metadata, stripping URLs and simplifying embed information.
- `create_combined_content` combines the parsed embed data with message text.

**nlp_utils.py**

- `add_ner_columns` adds named entity recognition columns to a DataFrame.
- `add_nounchunk_columns` extracts noun chunks for the specified field.

**mapping.py**

- Provides helpers for tokenisation and Jaccard similarity.
 - Includes functions for text chunking and embedding via `make_embedding`.
- Supports dimensionality reduction, clustering and topic modelling utilities.
- Contains helpers for handling CSV, PDF and zipped input files.

Import functions from these modules to build your own scripts.

A small example pipeline is provided in `example_pipeline.ipynb` to illustrate how the package can be used.
