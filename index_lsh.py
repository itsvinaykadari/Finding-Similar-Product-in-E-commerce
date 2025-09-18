# index_lsh.py
# This script will be used for building MinHash and LSH index for the dataset.

import json
from datasketch import MinHash, MinHashLSH
import re

def get_shingles(text, k=5):
	"""Generate k-character shingles from text."""
	text = text.lower()
	return set([text[i:i+k] for i in range(len(text)-k+1)])

def create_minhash(shingles, num_perm=128):
	m = MinHash(num_perm=num_perm)
	for shingle in shingles:
		m.update(shingle.encode('utf8'))
	return m

def build_lsh_index(products, field='title', k=5, num_perm=128, threshold=0.5):
	lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
	minhashes = {}
	for i, prod in enumerate(products):
		text = prod.get(field, '')
		shingles = get_shingles(text, k)
		m = create_minhash(shingles, num_perm)
		lsh.insert(str(i), m)
		minhashes[str(i)] = m
	return lsh, minhashes

if __name__ == "__main__":
	# Load processed products
	with open("processed_appliances.json", "r", encoding="utf-8") as f:
		products = json.load(f)

	# Build LSH index for titles
	print("Building LSH index for product titles...")
	lsh_title, minhashes_title = build_lsh_index(products, field='title', k=5, num_perm=128, threshold=0.5)
	print(f"LSH index for titles built with {len(minhashes_title)} items.")

	# Build LSH index for descriptions
	print("Building LSH index for product descriptions...")
	lsh_desc, minhashes_desc = build_lsh_index(products, field='description', k=5, num_perm=128, threshold=0.5)
	print(f"LSH index for descriptions built with {len(minhashes_desc)} items.")

	# Optionally, save the LSH/minhash objects using pickle for later use
	import pickle
	with open("lsh_title.pkl", "wb") as f:
		pickle.dump((lsh_title, minhashes_title), f)
	with open("lsh_desc.pkl", "wb") as f:
		pickle.dump((lsh_desc, minhashes_desc), f)
