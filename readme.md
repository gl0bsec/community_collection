# Community Collection

This repository contains utilities for processing and analysing text data.

The original notebooks have been refactored into a small Python package so
pipelines can be built programmatically without relying on notebooks.

## Package layout

```
community_collection/
    __init__.py
    parsing.py         # message parsing helpers
    nlp_utils.py       # spaCy based NLP helpers
    mapping.py         # vectorisation, clustering and topic modelling
```

Import functions from these modules to build your own scripts.
