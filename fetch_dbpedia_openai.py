"""Download DBpedia-OpenAI-1M (ada-002, 1536-D) and write dbpedia_openai_1M.npy.

Streams the 26 parquet shards (~9.5 GB) and keeps only the embedding
column as float32 (1M x 1536 = 6.1 GB npy). Text columns are dropped.
"""

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "KShivendu/dbpedia-entities-openai-1M"
OUT = "dbpedia_openai_1M.npy"

shards = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                if f.endswith(".parquet"))
print(f"{len(shards)} shards")

chunks = []
for i, name in enumerate(shards):
    path = hf_hub_download(REPO, name, repo_type="dataset")
    col = pq.read_table(path, columns=["openai"]).column("openai").combine_chunks()
    chunks.append(np.asarray(col.values, dtype=np.float32).reshape(len(col), -1))
    print(f"  shard {i + 1}/{len(shards)}: {chunks[-1].shape}", flush=True)

X = np.vstack(chunks)
print("total:", X.shape, X.dtype)
np.save(OUT, X)
print("saved", OUT)
