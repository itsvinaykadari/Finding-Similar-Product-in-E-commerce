# query.py
# This script will contain functions to query similar items using the LSH index.

import json
import pickle
from index_lsh import get_shingles, create_minhash

def load_indexes():
	with open("lsh_title.pkl", "rb") as f:
		lsh_title, minhashes_title = pickle.load(f)
	with open("lsh_desc.pkl", "rb") as f:
		lsh_desc, minhashes_desc = pickle.load(f)
	with open("processed_appliances.json", "r", encoding="utf-8") as f:
		products = json.load(f)
	return lsh_title, minhashes_title, lsh_desc, minhashes_desc, products

def query_similar_products(product_idx, mode='PST', top_k=10, k=5, num_perm=128):
	"""
	Query similar products for a given product index.
	mode: 'PST' (title), 'PSD' (description), 'PSTD' (title+description)
	"""
	lsh_title, minhashes_title, lsh_desc, minhashes_desc, products = load_indexes()
	prod = products[product_idx]
	if mode == 'PST':
		query_text = prod['title']
		lsh = lsh_title
		minhashes = minhashes_title
	elif mode == 'PSD':
		query_text = prod['description']
		lsh = lsh_desc
		minhashes = minhashes_desc
	elif mode == 'PSTD':
		query_text = prod['title'] + ' ' + prod['description']
		# For PSTD, create a new MinHash on the fly
		shingles = get_shingles(query_text, k)
		m = create_minhash(shingles, num_perm)
		# Use both indexes and merge results (simple union for now)
		result_title = set(lsh_title.query(m))
		result_desc = set(lsh_desc.query(m))
		result = list(result_title.union(result_desc))
		# Score by Jaccard similarity
		scored = []
		for idx in result:
			idx = int(idx)
			cand_text = products[idx]['title'] + ' ' + products[idx]['description']
			cand_shingles = get_shingles(cand_text, k)
			cand_m = create_minhash(cand_shingles, num_perm)
			score = m.jaccard(cand_m)
			scored.append((idx, score))
		scored.sort(key=lambda x: x[1], reverse=True)
		return [i for i, _ in scored[:top_k]]
	else:
		raise ValueError("Invalid mode. Use 'PST', 'PSD', or 'PSTD'.")

	# For PST or PSD
	shingles = get_shingles(query_text, k)
	m = create_minhash(shingles, num_perm)
	result = lsh.query(m)
	# Score by Jaccard similarity
	scored = []
	for idx in result:
		idx = int(idx)
		if mode == 'PSD':
			cand_text = products[idx]['description']
		elif mode == 'PST':
			cand_text = products[idx]['title']
		else: # Should not happen given the checks above, but as a fallback
			continue # Or raise an error
		cand_shingles = get_shingles(cand_text, k)
		cand_m = create_minhash(cand_shingles, num_perm)
		score = m.jaccard(cand_m)
		scored.append((idx, score))
	scored.sort(key=lambda x: x[1], reverse=True)
	return [i for i, _ in scored[:top_k]]

if __name__ == "__main__":
	# Example: Query similar products for the first product using all three modes
	for mode in ['PST', 'PSD', 'PSTD']:
		print(f"\nTop 5 similar products for product 0 using {mode}:")
		result = query_similar_products(0, mode=mode, top_k=5)
		print(result)
