import os
import re
import json
import math
import zipfile
import itertools
from typing import Iterable, Generator, Tuple, List

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, BertModel
import torch
from pacmap import PaCMAP
import fitz
import semchunk
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords


def tokenize(text: str) -> set[str]:
    text_no_urls = re.sub(r"https?://\S+|www\.\S+", "", text)
    text_clean = " ".join(text_no_urls.split()).lower()
    return set(re.findall(r"\b\w+\b", text_clean))


def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    union = set1 | set2
    if not union:
        return 0.0
    inter = set1 & set2
    return len(inter) / len(union)


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_filter_input(file_path: str, text_column: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(file_path)
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    filtered_df = df[df[text_column].str.strip().astype(bool)].copy()
    texts = filtered_df[text_column].tolist()
    return filtered_df, texts


def chunk(
    texts: List[str],
    model: str = "avsolatorio/GIST-large-Embedding-v0",
    size: int | None = None,
    max_token_chars: int | None = None,
    processes: int = 1,
) -> List[List[str]]:
    tokenizer = AutoTokenizer.from_pretrained(model, model_max_length=512)
    if size is None:
        size = int(tokenizer.model_max_length * 0.9)

    chunker = semchunk.chunkerify(tokenizer, chunk_size=size, max_token_chars=max_token_chars)
    return [chunker(text, processes=processes) for text in texts]


def batch_generator(iterable: Iterable, batch_size: int) -> Generator[List, None, None]:
    iterator = iter(iterable)
    for first in iterator:
        yield list(itertools.chain([first], itertools.islice(iterator, batch_size - 1)))


def make_embedding(
    texts: List[List[str]],
    model: str = "intfloat/multilingual-e5-large-instruct",
    normalise: bool = True,
    batch_size: int = 192,
    gpu: bool = True,
    progress: bool = True,
    return_chunk_vectors: bool = False,
    instruction_prefix: str | None = None,
) -> List[List[float]] | Tuple[List[List[float]], List[List[float]]]:
    """Generate document embeddings and optionally chunk embeddings.

    Parameters mirror the previous ``vectorise`` and ``improved_vectorise``
    functions. ``texts`` should be a list of lists where each sub list contains
    the chunks for a document. When ``instruction_prefix`` is provided each
    chunk is prefixed with that text before embedding. Set
    ``return_chunk_vectors`` to ``True`` to also return vectors for each chunk.
    """

    model_instance = AutoModel.from_pretrained(model)
    tokeniser = AutoTokenizer.from_pretrained(model)
    if gpu and torch.cuda.is_available():
        model_instance = model_instance.to("cuda")
    else:
        gpu = False
        if torch.cuda.device_count() == 0:
            print("GPU not available, using CPU instead")

    all_chunks: List[str] = []
    boundaries: List[Tuple[int, int]] = []
    start = 0
    for text_chunks in texts:
        if instruction_prefix is not None:
            processed = [instruction_prefix + chunk for chunk in text_chunks]
        else:
            processed = list(text_chunks)
        all_chunks.extend(processed)
        boundaries.append((start, start + len(text_chunks)))
        start += len(text_chunks)

    chunk_vectors: List[List[float]] = []
    with tqdm(total=len(all_chunks), disable=not progress, unit=" chunk") as bar:
        for batch in batch_generator(all_chunks, batch_size):
            inputs = tokeniser(
                batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            if gpu and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model_instance(**inputs)
                batch_vec = outputs[0][:, 0]
                if normalise:
                    batch_vec = torch.nn.functional.normalize(batch_vec, p=2, dim=1)
                batch_vec = batch_vec.cpu()
                chunk_vectors.extend(batch_vec.tolist())
            bar.update(len(batch))

    doc_vectors: List[List[float]] = []
    for start, end in boundaries:
        doc_chunks = torch.tensor(chunk_vectors[start:end])
        doc_vector = torch.mean(doc_chunks, dim=0).tolist()
        doc_vectors.append(doc_vector)

    if return_chunk_vectors:
        return doc_vectors, chunk_vectors
    return doc_vectors


def reduce_vectors(
    vectors: List[List[float]],
    clusterisable_dimensions: int = 80,
    map_dimensions: int = 2,
) -> Tuple[List[List[float]], List[List[float]]]:
    config = dict(n_neighbors=None, apply_pca=False, save_tree=True, verbose=True)
    cluster_model = PaCMAP(n_components=clusterisable_dimensions, **config)
    map_model = PaCMAP(n_components=map_dimensions, **config)
    clusterisable_vectors = cluster_model.fit_transform(vectors).tolist()
    map_vectors = map_model.fit_transform(vectors).tolist()
    return clusterisable_vectors, map_vectors


def csv_column_to_list(file_path: str, column_name: str) -> List[str]:
    df = pd.read_csv(file_path)
    return df[column_name].dropna().astype(str).tolist()


def save_reduced_vectors(clusterisable_vectors: List[List[float]], map_vectors: List[List[float]], clusterisable_filename: str, map_filename: str) -> None:
    pd.DataFrame(clusterisable_vectors).to_csv(clusterisable_filename, index=False)
    pd.DataFrame(map_vectors).to_csv(map_filename, index=False)


def preprocess_text(text: str) -> str:
    return text


def load_documents(folder_path: str) -> pd.DataFrame:
    data = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
                preprocessed_text = preprocess_text(raw_text)
                data.append((file, preprocessed_text))
        elif file.endswith(".pdf"):
            try:
                pdf_text = []
                with fitz.open(file_path) as pdf:
                    for page in pdf:
                        pdf_text.append(page.get_text())
                raw_text = " ".join(pdf_text)
                preprocessed_text = preprocess_text(raw_text)
                data.append((file, preprocessed_text))
            except Exception as e:
                print(f"Error processing PDF {file}: {e}")
    df = pd.DataFrame(data, columns=["filename", "content"])
    return df


def unzip_files(zip_filepath: str, extract_to_path: str) -> None:
    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f"Files extracted successfully to: {extract_to_path}")
    except FileNotFoundError:
        print(f"Error: Zip file not found at {zip_filepath}")
    except zipfile.BadZipFile:
        print(f"Error: Invalid zip file at {zip_filepath}")




def preprocess_dataframe_columns(
    df: pd.DataFrame,
    content_column: str,
    filename_column: str,
    cluster_column: str | None = None,
) -> Tuple[List[str], List[str], List[str]]:
    def preproc(text: str) -> str:
        text = text.encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    processed_content = df[content_column].dropna().astype(str).apply(preproc).tolist()
    filenames = df[filename_column].dropna().astype(str).tolist()
    clusters: List[str] = []
    if cluster_column is not None and cluster_column in df.columns:
        clusters = df[cluster_column].dropna().astype(str).tolist()
    return processed_content, filenames, clusters


def chunk_with_metadata(
    texts: List[str],
    filenames: List[str],
    clusters: List[str] | None = None,
    model: str = "intfloat/multilingual-e5-large-instruct",
    size: int | None = None,
    max_token_chars: int | None = None,
    processes: int = 1,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    tokenizer = AutoTokenizer.from_pretrained(model, model_max_length=512)
    if size is None:
        size = int(tokenizer.model_max_length * 0.6)
    chunker = semchunk.chunkerify(tokenizer, chunk_size=size, max_token_chars=max_token_chars)
    chunked_texts = []
    chunk_filenames = []
    chunk_clusters = []
    if not clusters:
        clusters = [""] * len(texts)
    elif len(clusters) < len(texts):
        clusters = clusters + [""] * (len(texts) - len(clusters))
    for text, filename, cluster in zip(texts, filenames, clusters):
        chunks = chunker(text, processes=processes)
        chunked_texts.append(chunks)
        chunk_filenames.append([filename] * len(chunks))
        chunk_clusters.append([cluster] * len(chunks))
    return chunked_texts, chunk_filenames, chunk_clusters


def save_chunk_embeddings_json(
    vectors: List,
    chunks: List[str],
    filenames: List[str],
    clusters: List[str],
    output_path: str,
) -> List[dict]:
    if isinstance(vectors, np.ndarray):
        vectors = vectors.tolist()
    data = []
    for i in range(len(chunks)):
        data.append({
            "chunk": chunks[i],
            "filename": filenames[i],
            "cluster": clusters[i],
            "length": len(chunks[i]),
            "embedding": vectors[i],
        })
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} chunk embeddings to {output_path}")
    return data


def cluster(
    vectors: List[List[float]],
    min_cluster_size: int | None = None,
    min_samples: int = 3,
) -> List[int]:
    vectors_np = np.array(vectors)
    if min_cluster_size is None:
        min_cluster_size = math.ceil(len(vectors_np) * 0.01)
        if min_cluster_size < 2:
            min_cluster_size = 2
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=int(math.log(len(vectors_np))))
    clusterer.fit(vectors_np)
    return clusterer.labels_.tolist()


def filter_vectors_by_text_name(doc_topics_df: pd.DataFrame, chunk_vectors_df: pd.DataFrame) -> pd.DataFrame:
    valid_text_names = doc_topics_df["text_name"].unique()
    filtered_df = chunk_vectors_df[chunk_vectors_df["text_name"].isin(valid_text_names)].copy()
    return filtered_df


def filter_topics_sequentially(
    chunk_df: pd.DataFrame,
    doc_cluster_df: pd.DataFrame,
    chunk_text_col: str = "text_name",
    chunk_topic_col: str = "topic_name",
    doc_text_col: str = "text_name",
    doc_cluster_col: str = "Cluster",
) -> Tuple[pd.DataFrame, dict]:
    topic_text_counts = chunk_df.groupby(chunk_topic_col)[chunk_text_col].nunique()
    total_unique_texts = chunk_df[chunk_text_col].nunique()
    if total_unique_texts == 0 or chunk_df.empty:
        empty_df = chunk_df.head(0).copy()
        if doc_cluster_col not in empty_df.columns:
            cluster_dtype = doc_cluster_df[doc_cluster_col].dtype if doc_cluster_col in doc_cluster_df.columns else object
            empty_df[doc_cluster_col] = pd.Series(dtype=cluster_dtype)
        return empty_df, {"initial": 0, "removed1": 0, "removed2": 0, "removed3": 0, "final_remaining": 0}

    topic_text_percentage = (topic_text_counts / total_unique_texts) * 100
    df_topic_texts_unique = chunk_df[[chunk_text_col, chunk_topic_col]].drop_duplicates()
    df_doc_clusters_unique = doc_cluster_df[[doc_text_col, doc_cluster_col]].dropna(subset=[doc_cluster_col]).drop_duplicates()
    df_doc_clusters_unique_renamed = df_doc_clusters_unique.rename(columns={doc_text_col: chunk_text_col, doc_cluster_col: "_Cluster_ID_Internal"})
    merged_topics_clusters = pd.merge(df_topic_texts_unique, df_doc_clusters_unique_renamed, on=chunk_text_col, how="inner")
    topic_doc_cluster_counts = merged_topics_clusters.groupby(chunk_topic_col)["_Cluster_ID_Internal"].nunique()

    all_topics_set = set(chunk_df[chunk_topic_col].unique())
    set1_one_text = set(topic_text_counts[topic_text_counts == 1].index)
    set2_less_than_2_percent = set(topic_text_percentage[topic_text_percentage < 2].index)
    set3_one_doc_cluster = set(topic for topic in all_topics_set if topic_doc_cluster_counts.get(topic, 0) == 1)

    filtering_stats = {"initial": len(all_topics_set)}
    filtered1 = all_topics_set.intersection(set1_one_text)
    removed1_count = len(filtered1)
    remaining_topics1 = all_topics_set - filtered1
    filtering_stats["removed1"] = removed1_count

    filtered2 = remaining_topics1.intersection(set2_less_than_2_percent)
    removed2_count = len(filtered2)
    remaining_topics2 = remaining_topics1 - filtered2
    filtering_stats["removed2"] = removed2_count

    filtered3 = remaining_topics2.intersection(set3_one_doc_cluster)
    removed3_count = len(filtered3)
    remaining_topics3 = remaining_topics2 - filtered3
    filtering_stats["removed3"] = removed3_count
    filtering_stats["final_remaining"] = len(remaining_topics3)

    df_filtered_chunks = chunk_df[chunk_df[chunk_topic_col].isin(remaining_topics3)].copy()
    doc_cluster_info = doc_cluster_df[[doc_text_col, doc_cluster_col]].drop_duplicates(subset=[doc_text_col])
    df_final = pd.merge(
        df_filtered_chunks,
        doc_cluster_info.rename(columns={doc_text_col: chunk_text_col}),
        on=chunk_text_col,
        how="left",
    )
    return df_final, filtering_stats


def json_to_dataframe(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for cluster, file_entries in data.items():
        for file_entry in file_entries:
            if isinstance(file_entry, dict):
                records.append({"cluster": cluster, "filename": file_entry["filename"], "content": file_entry["content"]})
    return pd.DataFrame(records)


def use_bertopic_with_custom_vectors(chunk_vectors: List[List[float]], documents: List[str], n_topics: int = 7):
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    english_stopwords = stopwords.words("english")
    additional_stopwords = [
        "maximum", "characters", "said", "also", "would", "could", "may", "might", "like",
        "many", "much", "get", "well", "even", "still", "back", "see", "way", "thing", "make",
        "made", "got", "go", "going",
    ]
    extended_stopwords = list(set(english_stopwords + additional_stopwords))

    preprocessed_documents = []
    for doc in documents:
        text = doc.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        preprocessed_documents.append(text)

    embeddings = np.array(chunk_vectors)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=27, metric="euclidean", cluster_selection_method="eom", prediction_data=True, min_samples=2)
    vectorizer = CountVectorizer(stop_words=extended_stopwords, ngram_range=(1, 2), min_df=0.10, max_df=0.90)

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=n_topics,
        min_topic_size=5,
        top_n_words=10,
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(preprocessed_documents, embeddings=embeddings)
    return topic_model, topics
