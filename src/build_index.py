import os, glob
import numpy as np
import faiss

PROC_DIR = "../data/processed"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 모든 임베딩 불러와서 스택
embs = []
for npz in glob.glob(os.path.join(PROC_DIR, "*.npz")):
    d = np.load(npz)
    embs.append(d['emb'])
embs = np.vstack(embs).astype('float32')

# FAISS L2 인덱스 생성
dim = embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs)
# 저장
faiss.write_index(index, os.path.join(MODEL_DIR, 'speaker.index'))
print("Saved speaker.index with", index.ntotal, "embeddings")