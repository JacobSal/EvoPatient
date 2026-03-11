import os
import sys
import json
from typing import List
from pathlib import Path

from dotenv import load_dotenv
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi
import numpy as np

# Allow importing local tools from the parent directory (relative path)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from helper_functions import *                  # noqa: F401,F403 needed: RecursiveCharacterTextSplitter, OpenAIEmbeddings, FAISS, show_context, replace_t_with_space_for_text
from evaluation.evalute_rag import *            # noqa: F401,F403

# ========= Load .env (do not hardcode keys/URLs in code) =========
load_dotenv()  # Place a .env in the project root containing OPENAI_API_KEY, OPENAI_API_BASE (optional)

# (Optional) base path / data file relative path
DATA_DIR = Path("../data")
PDF_PATH = DATA_DIR / "Understanding_Climate_Change.pdf"  # 如需使用PDF时可用；当前脚本未直接用到

# example text (relative path / environment agnostic)
content = (
    '1、患者，......，考虑转移。轻度鼻窦炎；左侧乳突炎。'
)


def encode_text_and_get_split_documents(content: str, chunk_size: int = 200, chunk_overlap: int = 50):
    """Split text -> clean -> build vector index"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_text(content)
    cleaned_texts = replace_t_with_space_for_text(texts)

    embeddings = OpenAIEmbeddings()  # 从环境变量读取 Key/Endpoint
    vectorstore = FAISS.from_texts(cleaned_texts, embeddings)

    return vectorstore, cleaned_texts


vectorstore, cleaned_texts = encode_text_and_get_split_documents(content)


def create_bm25_index(texts: List[str]) -> BM25Okapi:
    """基于分词后的文本列表创建 BM25 索引"""
    tokenized_docs = [doc.split() for doc in texts]
    return BM25Okapi(tokenized_docs)


bm25 = create_bm25_index(cleaned_texts)


def _safe_minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalization; if max == min return all zeros to avoid division by zero."""
    if arr.size == 0:
        return arr
    a_min, a_max = np.min(arr), np.max(arr)
    if a_max == a_min:
        return np.zeros_like(arr)
    return (arr - a_min) / (a_max - a_min)


def fusion_retrieval(
    vectorstore,
    bm25: BM25Okapi,
    query: str,
    all_texts: List[str],
    k: int = 5,
    alpha: float = 0.5
) -> List[Document]:
    """
    Keyword (BM25) + vector retrieval (FAISS) score fusion retrieval.
    alpha: weight for vector score; (1 - alpha) for BM25.
    """
    # 1) all documents (constructed from cleaned_texts)
    all_docs: List[Document] = [Document(page_content=t) for t in all_texts]

    # 2) BM25 scores (aligned with all_docs order)
    bm25_scores = bm25.get_scores(query.split())  # shape: (N,)

    # 3) 向量检索得分（与 all_docs 对齐：逐个检索计算距离）
    # Note: FAISS's similarity_search_with_score gives smaller distance for more similar; need to invert/normalize
    vec_scores = []
    for t in all_texts:
        # 使用每个片段做“查询文本”的方式计算距离可能效率较低，
        # 这里用真正的 query 来检索全量，再按内容对齐。
        pass

    # More efficient: get similarity results for the whole query at once, then align order
    vec_results = vectorstore.similarity_search_with_score(query, k=len(all_texts))
    # 建立 {内容: 距离} 映射（若重复片段，可考虑 (内容, idx) 做键）
    content2dist = {doc.page_content: dist for doc, dist in vec_results}
    # Align with all_texts order; assign a poor score for missing entries (e.g. max distance + 1)
    fallback = (max(content2dist.values()) + 1.0) if content2dist else 1.0
    vector_dists = np.array([content2dist.get(t, fallback) for t in all_texts], dtype=float)
    vector_scores = 1.0 - _safe_minmax_norm(vector_dists)  # 距离 -> 相似度

    # 4) normalize BM25
    bm25_scores = _safe_minmax_norm(np.array(bm25_scores, dtype=float))

    # 5) fuse scores
    combined_scores = alpha * vector_scores + (1.0 - alpha) * bm25_scores

    # 6) sort and return top-k
    sorted_indices = np.argsort(combined_scores)[::-1]
    top_indices = sorted_indices[:k]
    return [all_docs[i] for i in top_indices]


# 查询
query = "Have you recently had symptoms like coughing?"

# 融合检索
top_docs = fusion_retrieval(vectorstore, bm25, query, all_texts=cleaned_texts, k=5, alpha=0.5)
docs_content = [doc.page_content for doc in top_docs]
show_context(docs_content)
