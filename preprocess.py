# preprocess.py
# This script will contain functions for cleaning and processing the dataset.

import json
import re
from typing import List, Dict

def load_products(json_path: str) -> List[Dict]:
	"""Load products from a line-delimited JSON file."""
	products = []
	with open(json_path, 'r', encoding='utf-8') as f:
		for line in f:
			try:
				products.append(json.loads(line))
			except Exception:
				continue
	return products

def clean_text(text: str) -> str:
	"""Remove HTML tags and extra whitespace from text."""
	if not isinstance(text, str):
		return ''
	# Remove HTML tags
	text = re.sub(r'<.*?>', '', text)
	# Remove extra whitespace
	text = re.sub(r'\s+', ' ', text)
	return text.strip()

def preprocess_products(products: List[Dict]) -> List[Dict]:
	"""Clean and extract relevant fields from products."""
	processed = []
	for prod in products:
		# Handle description field which is an array
		description_raw = prod.get('description', [])
		if isinstance(description_raw, list) and description_raw:
			# Join all description strings with a space
			description_text = ' '.join(description_raw)
		elif isinstance(description_raw, str):
			description_text = description_raw
		else:
			description_text = ''
		
		item = {
			'asin': prod.get('asin', ''),
			'title': clean_text(prod.get('title', '')),
			'description': clean_text(description_text),
			'also_buy': prod.get('also_buy', []),
			'also_view': prod.get('also_view', []),
			'similar_item': prod.get('similar_item', [])
		}
		processed.append(item)
	return processed

if __name__ == "__main__":
	# Example usage
	products = load_products("meta_Appliances.json")
	processed = preprocess_products(products)
	print(f"Loaded {len(products)} products. Cleaned {len(processed)} products.")
	# Optionally, save processed data for later use
	with open("processed_appliances.json", "w", encoding="utf-8") as f:
		json.dump(processed, f, ensure_ascii=False, indent=2)
