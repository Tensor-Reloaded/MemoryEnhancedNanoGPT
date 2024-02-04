import numpy as np
import faiss  # make faiss available
import time  # Import the time module

d = 64  # embedding dimension
nb = 10_000_000  # database size
nc = 2_000  # number of centroids

number_of_queries = 61_440

top_k = 3
nprobe = 10  # number of neighbor clusters to look at

m = 8  # number of centroid IDs in final compressed vectors
bits = 8  # number of bits in each centroid

np.random.seed(123)

xb = np.random.random((nb, d)).astype('float32')
xt = np.random.random((nb, d)).astype('float32')
xq = np.random.random((number_of_queries, d)).astype('float32')



# Time the indexing operation
start_time = time.time()

quantizer = faiss.IndexFlatL2(d)  # we keep the same L2 distance flat index
# index = faiss.IndexIVFFlat(quantizer, d, nc, faiss.METRIC_L2)
index = faiss.IndexIVFPQ(quantizer, d, nc, m, bits)
index.train(xt)
index.add(xb)
index.nprobe = nprobe

indexing_duration = time.time() - start_time
print("Index constructed", index.ntotal)
print(f"Index constructed duration: {indexing_duration} seconds")

# Time the search operation
start_time = time.time()
D, I = index.search(xq, top_k)
search_duration = time.time() - start_time
print("Search done")
print(f"Search duration: {search_duration} seconds")


index.make_direct_map()

indices = np.unique([item for sublist in I for item in sublist])
print(indices)


# Time the update operation
start_time = time.time()
index.update_vectors(indices, xb[indices].copy() * 0.5)
update_duration = time.time() - start_time
print(f"Update duration: {update_duration} seconds")
