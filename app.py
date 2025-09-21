# app.py
# Flask UI for interacting with the dataset and LSH index.

from flask import Flask, render_template_string, request, jsonify
import json
import re
import numpy as np
from collections import Counter
from query import query_similar_products

app = Flask(__name__)

# Load products once at startup
with open("processed_appliances.json", "r", encoding="utf-8") as f:
        products = json.load(f)

# Create ASIN to index mapping for ground truth lookup
asin_to_idx = {prod['asin']: i for i, prod in enumerate(products)}

PRODUCTS_PER_PAGE = 20

def normalize_text(text):
    """Normalize text for better ROUGE scoring"""
    if not text:
        return ""
    
    # Convert to lowercase and remove extra whitespace
    text = text.lower().strip()
    
    # Remove punctuation and special characters
    import string
    text = ''.join(char if char not in string.punctuation else ' ' for char in text)
    
    # Remove common stopwords that don't add meaning
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
                'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'it', 'its', 'he', 'she',
                'they', 'we', 'you', 'i', 'me', 'him', 'her', 'them', 'us'}
    
    words = [word for word in text.split() if word and word not in stopwords]
    return ' '.join(words)

def calculate_rouge_1(predicted_text, reference_text):
    """Calculate ROUGE-1 score (unigram overlap) with normalization"""
    if not predicted_text or not reference_text:
        return 0.0
    
    pred_words = normalize_text(predicted_text).split()
    ref_words = normalize_text(reference_text).split()
    
    if not ref_words:
        return 0.0
    
    pred_counter = Counter(pred_words)
    ref_counter = Counter(ref_words)
    
    overlap = sum((pred_counter & ref_counter).values())
    return overlap / len(ref_words)

def calculate_rouge_2(predicted_text, reference_text):
    """Calculate ROUGE-2 score (bigram overlap) with normalization"""
    if not predicted_text or not reference_text:
        return 0.0
    
    def get_bigrams(words):
        return [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    
    pred_words = normalize_text(predicted_text).split()
    ref_words = normalize_text(reference_text).split()
    
    if len(ref_words) < 2:
        return 0.0
    
    pred_bigrams = get_bigrams(pred_words)
    ref_bigrams = get_bigrams(ref_words)
    
    if not ref_bigrams:
        return 0.0
    
    pred_counter = Counter(pred_bigrams)
    ref_counter = Counter(ref_bigrams)
    
    overlap = sum((pred_counter & ref_counter).values())
    return overlap / len(ref_bigrams)

def calculate_rouge_l(predicted_text, reference_text):
    """Calculate ROUGE-L score (Longest Common Subsequence) with normalization"""
    if not predicted_text or not reference_text:
        return 0.0
    
    def lcs_length(X, Y):
        m, n = len(X), len(Y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i-1] == Y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    pred_words = normalize_text(predicted_text).split()
    ref_words = normalize_text(reference_text).split()
    
    if not ref_words:
        return 0.0
    
    lcs_len = lcs_length(pred_words, ref_words)
    return lcs_len / len(ref_words)

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity using simple word overlap with synonyms"""
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    # Direct word overlap
    direct_overlap = len(words1 & words2)
    
    # Synonym/related word matching
    # Common electrical/appliance synonyms and related terms
    synonyms = {
        'amp': {'ampere', 'amperage', 'current'},
        'volt': {'voltage', 'volts', 'v'},
        'wire': {'cable', 'cord', 'wiring'},
        'outlet': {'receptacle', 'socket', 'plug'},
        'electrical': {'electric', 'power', 'energy'},
        'mount': {'mounting', 'installation', 'install'},
        'range': {'stove', 'oven', 'cooktop'},
        'dryer': {'laundry', 'clothes'},
        'black': {'blk', 'dark'},
        'white': {'wht', 'light'},
        'gauge': {'awg', 'size'},
        'heavy': {'duty', 'industrial', 'commercial'},
        'extension': {'ext', 'extender'},
        'indoor': {'interior', 'inside'},
        'outdoor': {'exterior', 'outside', 'weather'},
    }
    
    # Count synonym matches
    synonym_matches = 0
    for word1 in words1:
        for word2 in words2:
            if word1 != word2:  # Not direct match
                # Check if they're synonyms
                for key, syn_set in synonyms.items():
                    if (word1 == key and word2 in syn_set) or (word2 == key and word1 in syn_set) or (word1 in syn_set and word2 in syn_set):
                        synonym_matches += 1
                        break
    
    # Calculate semantic score
    total_overlap = direct_overlap + (synonym_matches * 0.7)  # Weight synonyms slightly lower
    max_possible = max(len(words1), len(words2))
    
    return total_overlap / max_possible if max_possible > 0 else 0.0

def calculate_rouge_scores(predicted_products, ground_truth_products):
    """Calculate ROUGE scores using optimized pairwise comparison and averaging"""
    if not predicted_products or not ground_truth_products:
        return {
            'rouge_1_title': 0.0,
            'rouge_2_title': 0.0,
            'rouge_l_title': 0.0,
            'rouge_1_desc': 0.0,
            'rouge_2_desc': 0.0,
            'rouge_l_desc': 0.0,
            'rouge_1_combined': 0.0,
            'rouge_2_combined': 0.0,
            'rouge_l_combined': 0.0,
            'semantic_similarity_title': 0.0,
            'semantic_similarity_combined': 0.0,
            'has_descriptions': False
        }
    
    # Pairwise comparison approach: compare each predicted against all ground truth
    def calculate_pairwise_rouge(pred_texts, ref_texts, rouge_func):
        """Calculate average ROUGE score using pairwise comparison"""
        if not pred_texts or not ref_texts:
            return 0.0
        
        total_score = 0.0
        comparison_count = 0
        
        for pred_text in pred_texts:
            if pred_text.strip():  # Only compare non-empty texts
                max_score = 0.0
                for ref_text in ref_texts:
                    if ref_text.strip():
                        score = rouge_func(pred_text, ref_text)
                        max_score = max(max_score, score)
                
                if max_score > 0:  # Only count valid comparisons
                    total_score += max_score
                    comparison_count += 1
        
        return total_score / comparison_count if comparison_count > 0 else 0.0
    
    # Extract texts for comparison
    pred_titles = [prod.get('title', '') for prod in predicted_products]
    ref_titles = [prod.get('title', '') for prod in ground_truth_products]
    
    pred_descs = [prod.get('description', '') for prod in predicted_products]
    ref_descs = [prod.get('description', '') for prod in ground_truth_products]
    
    # Combined title + description texts
    pred_combined = [
        (prod.get('title', '') + ' ' + prod.get('description', '')).strip() 
        for prod in predicted_products
    ]
    ref_combined = [
        (prod.get('title', '') + ' ' + prod.get('description', '')).strip() 
        for prod in ground_truth_products
    ]
    
    # Check if descriptions are actually available
    has_descriptions = any(desc.strip() for desc in pred_descs + ref_descs)
    
    return {
        'rouge_1_title': calculate_pairwise_rouge(pred_titles, ref_titles, calculate_rouge_1),
        'rouge_2_title': calculate_pairwise_rouge(pred_titles, ref_titles, calculate_rouge_2),
        'rouge_l_title': calculate_pairwise_rouge(pred_titles, ref_titles, calculate_rouge_l),
        'rouge_1_desc': calculate_pairwise_rouge(pred_descs, ref_descs, calculate_rouge_1) if has_descriptions else 0.0,
        'rouge_2_desc': calculate_pairwise_rouge(pred_descs, ref_descs, calculate_rouge_2) if has_descriptions else 0.0,
        'rouge_l_desc': calculate_pairwise_rouge(pred_descs, ref_descs, calculate_rouge_l) if has_descriptions else 0.0,
        'rouge_1_combined': calculate_pairwise_rouge(pred_combined, ref_combined, calculate_rouge_1),
        'rouge_2_combined': calculate_pairwise_rouge(pred_combined, ref_combined, calculate_rouge_2),
        'rouge_l_combined': calculate_pairwise_rouge(pred_combined, ref_combined, calculate_rouge_l),
        'semantic_similarity_title': calculate_pairwise_rouge(pred_titles, ref_titles, calculate_semantic_similarity),
        'semantic_similarity_combined': calculate_pairwise_rouge(pred_combined, ref_combined, calculate_semantic_similarity),
        'has_descriptions': has_descriptions
    }

def get_ground_truth_products(product_idx):
    """Get ground truth similar products for a given product index"""
    product = products[product_idx]
    similar_items = product.get('similar_item', [])
    
    # Handle different data types and extract ASINs from HTML
    asins = []
    
    if isinstance(similar_items, str) and similar_items.strip():
        # Extract ASINs from HTML using regex pattern for Amazon product URLs
        # Look for patterns like /dp/ASIN/ in href attributes
        asin_pattern = r'/dp/([A-Z0-9]{10})'
        matches = re.findall(asin_pattern, similar_items)
        asins = list(set(matches))  # Remove duplicates
    elif isinstance(similar_items, list):
        for item in similar_items:
            if isinstance(item, str) and item.strip():
                # Try to extract ASIN if it's HTML
                asin_pattern = r'/dp/([A-Z0-9]{10})'
                matches = re.findall(asin_pattern, item)
                asins.extend(matches)
            elif isinstance(item, str) and len(item) == 10 and item.isalnum():
                # Direct ASIN
                asins.append(item)
    
    # Convert ASINs to product indices
    ground_truth_indices = []
    for asin in asins:
        if asin in asin_to_idx:
            ground_truth_indices.append(asin_to_idx[asin])
    
    return ground_truth_indices

def calculate_precision_at_k(product_idx, mode, k):
    """Calculate precision@k for the given product and mode"""
    try:
        # Get LSH results
        lsh_results = query_similar_products(product_idx, mode=mode, top_k=k)
        lsh_results = [idx for idx in lsh_results if idx != product_idx][:k]
        
        # Get ground truth
        ground_truth_indices = get_ground_truth_products(product_idx)
        
        if not ground_truth_indices:
            return None
        
        # Calculate intersection
        relevant_count = len(set(lsh_results) & set(ground_truth_indices))
        
        precision = relevant_count / k if k > 0 else 0.0
        return precision
    except Exception as e:
        print(f"Error calculating precision for product {product_idx}, mode {mode}, k {k}: {e}")
        return None

def calculate_all_metrics(product_idx, mode, k=10):
    """Calculate both precision@k and ROUGE scores for a given product and mode"""
    try:
        # Use higher k for ROUGE (more products = better ROUGE scores)
        rouge_k = max(k, 20)  # Use at least 20 products for ROUGE calculation
        
        # Get LSH results
        lsh_results = query_similar_products(product_idx, mode=mode, top_k=rouge_k)
        lsh_results = [idx for idx in lsh_results if idx != product_idx][:rouge_k]
        
        # Get ground truth
        ground_truth_indices = get_ground_truth_products(product_idx)
        
        if not ground_truth_indices:
            return None
        
        # Calculate precision@k for different k values (use original k)
        lsh_precision_results = lsh_results[:k]  # Only use top-k for precision
        precisions = {}
        for pk in [1, 3, 5, 10]:
            if pk <= len(lsh_precision_results):
                relevant_count = len(set(lsh_precision_results[:pk]) & set(ground_truth_indices))
                precisions[f'p@{pk}'] = relevant_count / pk if pk > 0 else 0.0
            else:
                precisions[f'p@{pk}'] = None
        
        # Get product objects for ROUGE calculation (use all rouge_k results)
        predicted_products = [products[i] for i in lsh_results]
        ground_truth_products = [products[i] for i in ground_truth_indices]
        
        # Calculate ROUGE scores
        rouge_scores = calculate_rouge_scores(predicted_products, ground_truth_products)
        
        return {
            'precisions': precisions,
            'rouge_scores': rouge_scores
        }
    except Exception as e:
        print(f"Error calculating metrics for product {product_idx}, mode {mode}, k {k}: {e}")
        return None

def calculate_all_precisions(product_idx, mode):
    """Calculate precision@k for k=1,3,5,10"""
    precisions = {}
    for k in [1, 3, 5, 10]:
        precisions[f'p@{k}'] = calculate_precision_at_k(product_idx, mode, k)
    return precisions

def calculate_precisions_and_rouge_for_all_modes(product_idx):
    """Calculate precision@k and ROUGE scores for all modes (PST, PSD, PSTD)"""
    all_metrics = {}
    modes = ['PST', 'PSD', 'PSTD']
    
    # Only calculate if product has ground truth
    product = products[product_idx]
    if not (product.get('similar_item') and product['similar_item']):
        return None
    
    for mode in modes:
        metrics = calculate_all_metrics(product_idx, mode, k=10)
        if metrics:
            all_metrics[mode] = metrics
        else:
            all_metrics[mode] = {
                'precisions': {f'p@{k}': None for k in [1, 3, 5, 10]},
                'rouge_scores': {
                    'rouge_1_title': 0.0,
                    'rouge_2_title': 0.0,
                    'rouge_l_title': 0.0,
                    'rouge_1_desc': 0.0,
                    'rouge_2_desc': 0.0,
                    'rouge_l_desc': 0.0,
                    'rouge_1_combined': 0.0,
                    'rouge_2_combined': 0.0,
                    'rouge_l_combined': 0.0,
                    'semantic_similarity_title': 0.0,
                    'semantic_similarity_combined': 0.0,
                    'has_descriptions': False
                }
            }
    
    return all_metrics


HOME_TEMPLATE = '''
<style>
body { font-family: Arial, sans-serif; background: #f7f7f7; }
.main-container { max-width: 1200px; margin: 32px auto; background: #fff; box-shadow: 0 2px 8px #ccc; border-radius: 12px; padding: 32px; }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 32px; }
.prod-card { background: #fafafa; border-radius: 10px; box-shadow: 0 1px 4px #ddd; padding: 18px; display: flex; flex-direction: column; align-items: center; min-height: 600px; }
.prod-img { max-width: 100px; max-height: 100px; border-radius: 8px; box-shadow: 0 1px 4px #bbb; margin-bottom: 12px; }
.prod-title { font-size: 1.1em; font-weight: bold; margin-bottom: 8px; color: #2a2a2a; text-align: center; }
.prod-price { color: #007600; font-weight: bold; margin-bottom: 8px; }
.prod-link { color: #0056b3; text-decoration: none; font-size: 1em; margin-bottom: 8px; }
.sim-options { width: 100%; margin-top: 10px; }
.sim-row { display: flex; flex-direction: row; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.sim-label { font-size: 0.98em; color: #444; }
.sim-btn { background: #0056b3; color: #fff; border: none; border-radius: 5px; padding: 6px 14px; font-size: 0.98em; cursor: pointer; margin-left: 8px; }
.sim-btn:hover { background: #003d80; }
.topk-form { margin-top: 10px; }
.no-sim { color: #c00; font-size: 0.98em; margin-top: 10px; }
.pagination { margin-top: 32px; text-align: center; }
.pagination a { color: #0056b3; text-decoration: none; font-weight: bold; margin: 0 12px; }
.pagination a:hover { text-decoration: underline; }
.precision-box { background: #e8f4f8; border-radius: 6px; padding: 8px; margin-top: 8px; font-size: 0.85em; }
.precision-row { display: flex; justify-content: space-between; margin-bottom: 3px; }
.precision-label { font-weight: bold; }
.precision-value { color: #0066cc; }
.rouge-box { background: #f0e8f8; border-radius: 6px; padding: 8px; margin-top: 8px; font-size: 0.85em; }
.rouge-row { display: flex; justify-content: space-between; margin-bottom: 3px; }
.rouge-label { font-weight: bold; }
.rouge-value { color: #8000ff; }
.metrics-container { margin-top: 10px; }
.metrics-toggle { background: #17a2b8; color: white; border: none; border-radius: 4px; padding: 4px 8px; font-size: 0.8em; cursor: pointer; margin-top: 5px; }
.metrics-toggle:hover { background: #138496; }
.no-precision { background: #f0f0f0; color: #666; font-style: italic; }
</style>
<div class="main-container">
    <h1 style="text-align: center; margin-bottom: 32px; color: #2c3e50; font-size: 2.5em; font-weight: bold;">Finding Similar Product in E-commerce</h1>
    <h2 style="margin-bottom:24px;">Product List (Page {{ page }})</h2>
    <div style="margin-bottom: 20px;">
        <a href="/exercise3" style="background: #e74c3c; color: white; text-decoration: none; padding: 12px 25px; border-radius: 8px; font-weight: bold;">
            Exercise 3: LSH Evaluation
        </a>
    </div>
    <form class="topk-form" method="get" action="/">
        <label for="top_k">Show top-k similar products (k): </label>
        <input type="number" id="top_k" name="top_k" min="1" max="20" value="{{ top_k }}" style="width:60px;">
        <button type="submit" class="sim-btn">Apply</button>
    </form>
    <div class="grid">
    {% for idx, prod, precisions in products %}
        <div class="prod-card">
            {% set img = prod.get('imageURLHighRes', []) or prod.get('imageURL', []) %}
            {% if img and img[0] %}
                <img src="{{ img[0] }}" alt="Product Image" class="prod-img">
            {% endif %}
            <div class="prod-title">{{ prod['title'] }}</div>
            {% if prod.get('price') %}<div class="prod-price">{{ prod['price'] }}</div>{% endif %}
            <a class="prod-link" href="https://www.amazon.com/dp/{{ prod['asin'] }}" target="_blank">View on Amazon</a>
            <div class="sim-options">
                <div class="sim-row">
                    <span class="sim-label">Similar Title</span>
                    <form action="/similar" method="get" style="display:inline;">
                        <input type="hidden" name="product_idx" value="{{ idx }}">
                        <input type="hidden" name="mode" value="PST">
                        <input type="hidden" name="top_k" value="{{ top_k }}">
                        <button type="submit" class="sim-btn">Show</button>
                    </form>
                </div>
                
                <div class="sim-row">
                    <span class="sim-label">Similar Description</span>
                    <form action="/similar" method="get" style="display:inline;">
                        <input type="hidden" name="product_idx" value="{{ idx }}">
                        <input type="hidden" name="mode" value="PSD">
                        <input type="hidden" name="top_k" value="{{ top_k }}">
                        <button type="submit" class="sim-btn">Show</button>
                    </form>
                </div>
                
                <div class="sim-row">
                    <span class="sim-label">Title + Description</span>
                    <form action="/similar" method="get" style="display:inline;">
                        <input type="hidden" name="product_idx" value="{{ idx }}">
                        <input type="hidden" name="mode" value="PSTD">
                        <input type="hidden" name="top_k" value="{{ top_k }}">
                        <button type="submit" class="sim-btn">Show</button>
                    </form>
                </div>
            </div>
            {% if prod.get('similar_item') and prod['similar_item'] %}
                <div style="margin-top:10px; color:#888; font-size:0.95em;">Has {{ prod['similar_item']|length }} ground-truth similar products.</div>
                
                <!-- Individual Calculate Precision Button -->
                <div style="margin-top: 10px;">
                    <button onclick="calculateMetrics({{ idx }}, {{ top_k }})" class="sim-btn" style="background: #28a745;" id="calc-btn-{{ idx }}">
                        Calculate Metrics
                    </button>
                </div>
                
                <!-- Metrics Results (hidden initially) -->
                <div id="metrics-results-{{ idx }}" style="display: none;">
                    <div class="precision-box">
                        <strong>Title Precision@k (PST):</strong><br>
                        <div id="pst-results-{{ idx }}"></div>
                        <button class="metrics-toggle" onclick="toggleRouge('pst', {{ idx }})">Show ROUGE</button>
                        <div id="pst-rouge-{{ idx }}" style="display: none;" class="rouge-box">
                            <div id="pst-rouge-results-{{ idx }}"></div>
                        </div>
                    </div>
                    <div class="precision-box">
                        <strong>Description Precision@k (PSD):</strong><br>
                        <div id="psd-results-{{ idx }}"></div>
                        <button class="metrics-toggle" onclick="toggleRouge('psd', {{ idx }})">Show ROUGE</button>
                        <div id="psd-rouge-{{ idx }}" style="display: none;" class="rouge-box">
                            <div id="psd-rouge-results-{{ idx }}"></div>
                        </div>
                    </div>
                    <div class="precision-box">
                        <strong>Title+Desc Precision@k (PSTD):</strong><br>
                        <div id="pstd-results-{{ idx }}"></div>
                        <button class="metrics-toggle" onclick="toggleRouge('pstd', {{ idx }})">Show ROUGE</button>
                        <div id="pstd-rouge-{{ idx }}" style="display: none;" class="rouge-box">
                            <div id="pstd-rouge-results-{{ idx }}"></div>
                        </div>
                    </div>
                </div>
            {% else %}
                <div class="precision-box no-precision">No ground truth available</div>
            {% endif %}
        </div>
    {% endfor %}
    </div>
    <div class="pagination">
        {% if page > 1 %}<a href="/?page={{ page-1 }}&top_k={{ top_k }}">&#8592; Previous</a>{% endif %}
        {% if end_idx < total %}<a href="/?page={{ page+1 }}&top_k={{ top_k }}">Next &#8594;</a>{% endif %}
    </div>
</div>

<script>
function calculateMetrics(productIdx, topK) {
    const button = document.getElementById('calc-btn-' + productIdx);
    const resultsDiv = document.getElementById('metrics-results-' + productIdx);
    
    // Show loading state
    button.textContent = 'Calculating...';
    button.disabled = true;
    button.style.background = '#6c757d';
    
    // Make AJAX request
    fetch('/precision?product_idx=' + productIdx + '&top_k=' + topK)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.precisions && data.rouge_scores) {
                // Show results
                resultsDiv.style.display = 'block';
                
                // Update PST results
                updateModeResults('pst', productIdx, data.precisions.PST, data.rouge_scores.PST);
                
                // Update PSD results  
                updateModeResults('psd', productIdx, data.precisions.PSD, data.rouge_scores.PSD);
                
                // Update PSTD results
                updateModeResults('pstd', productIdx, data.precisions.PSTD, data.rouge_scores.PSTD);
                
                // Update button
                button.textContent = '✓ Calculated';
                button.style.background = '#28a745';
            } else {
                alert('Error calculating metrics: ' + (data.error || 'Unknown error'));
                button.textContent = 'Calculate Metrics';
                button.style.background = '#28a745';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error calculating metrics');
            button.textContent = 'Calculate Metrics';
            button.style.background = '#28a745';
        })
        .finally(() => {
            button.disabled = false;
        });
}

function updateModeResults(mode, productIdx, precisions, rougeScores) {
    // Update precision results
    const precisionDiv = document.getElementById(mode + '-results-' + productIdx);
    let precisionHtml = '';
    for (const [k, p] of Object.entries(precisions || {})) {
        const value = p !== null ? p.toFixed(3) : 'N/A';
        precisionHtml += '<span class="precision-label">' + k + ':</span> <span class="precision-value">' + value + '</span><br>';
    }
    precisionDiv.innerHTML = precisionHtml;
    
    // Update ROUGE results
    const rougeDiv = document.getElementById(mode + '-rouge-results-' + productIdx);
    let rougeHtml = '<strong>Title ROUGE:</strong><br>';
    rougeHtml += '<span class="rouge-label">ROUGE-1:</span> <span class="rouge-value">' + (rougeScores.rouge_1_title || 0).toFixed(3) + '</span><br>';
    rougeHtml += '<span class="rouge-label">ROUGE-2:</span> <span class="rouge-value">' + (rougeScores.rouge_2_title || 0).toFixed(3) + '</span><br>';
    rougeHtml += '<span class="rouge-label">ROUGE-L:</span> <span class="rouge-value">' + (rougeScores.rouge_l_title || 0).toFixed(3) + '</span><br><br>';
    
    if (rougeScores.has_descriptions) {
        rougeHtml += '<strong>Description ROUGE:</strong><br>';
        rougeHtml += '<span class="rouge-label">ROUGE-1:</span> <span class="rouge-value">' + (rougeScores.rouge_1_desc || 0).toFixed(3) + '</span><br>';
        rougeHtml += '<span class="rouge-label">ROUGE-2:</span> <span class="rouge-value">' + (rougeScores.rouge_2_desc || 0).toFixed(3) + '</span><br>';
        rougeHtml += '<span class="rouge-label">ROUGE-L:</span> <span class="rouge-value">' + (rougeScores.rouge_l_desc || 0).toFixed(3) + '</span><br><br>';
    } else {
        rougeHtml += '<strong>Description ROUGE:</strong><br>';
        rougeHtml += '<span style="color: #999; font-style: italic;">No descriptions available in dataset</span><br><br>';
    }
    
    rougeHtml += '<strong>Combined (Title+Desc) ROUGE:</strong><br>';
    rougeHtml += '<span class="rouge-label">ROUGE-1:</span> <span class="rouge-value">' + (rougeScores.rouge_1_combined || 0).toFixed(3) + '</span><br>';
    rougeHtml += '<span class="rouge-label">ROUGE-2:</span> <span class="rouge-value">' + (rougeScores.rouge_2_combined || 0).toFixed(3) + '</span><br>';
    rougeHtml += '<span class="rouge-label">ROUGE-L:</span> <span class="rouge-value">' + (rougeScores.rouge_l_combined || 0).toFixed(3) + '</span><br>';
    
    rougeDiv.innerHTML = rougeHtml;
}

function toggleRouge(mode, productIdx) {
    const rougeDiv = document.getElementById(mode + '-rouge-' + productIdx);
    const toggleBtn = event.target;
    
    if (rougeDiv.style.display === 'none') {
        rougeDiv.style.display = 'block';
        toggleBtn.textContent = 'Hide ROUGE';
    } else {
        rougeDiv.style.display = 'none';
        toggleBtn.textContent = 'Show ROUGE';
    }
}
</script>
'''

EXERCISE3_TEMPLATE = '''
<style>
body { font-family: Arial, sans-serif; background: #f7f7f7; margin: 0; padding: 0; }
.container { max-width: 1200px; margin: 32px auto; background: #fff; box-shadow: 0 2px 8px #ccc; border-radius: 12px; padding: 32px; }
.header { text-align: center; margin-bottom: 30px; }
.header h1 { color: #2c3e50; margin-bottom: 10px; }
.header p { color: #7f8c8d; font-size: 18px; }
.requirements { background: #ecf0f1; border-radius: 8px; padding: 20px; margin: 20px 0; }
.requirements h3 { color: #34495e; margin-top: 0; }
.requirements ul { color: #555; }
.requirements li { margin-bottom: 8px; }
.eval-section { background: #f8f9fa; border-radius: 8px; padding: 25px; margin: 20px 0; border: 1px solid #e0e0e0; }
.eval-btn { background: #e74c3c; color: white; border: none; border-radius: 8px; padding: 15px 30px; cursor: pointer; font-size: 18px; font-weight: bold; margin: 15px 10px; transition: all 0.3s; }
.eval-btn:hover { background: #c0392b; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
.eval-btn:disabled { background: #6c757d; cursor: not-allowed; transform: none; box-shadow: none; }
.status-box { background: #3498db; color: white; border-radius: 8px; padding: 15px; margin: 20px 0; font-weight: bold; text-align: center; }
.status-box.success { background: #27ae60; }
.status-box.error { background: #e74c3c; }
.results-section { background: #ffffff; border-radius: 8px; padding: 25px; margin: 20px 0; border: 2px solid #3498db; }
.results-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
.results-table th, .results-table td { border: 1px solid #ddd; padding: 12px; text-align: center; }
.results-table th { background-color: #3498db; color: white; font-weight: bold; }
.results-table tr:nth-child(even) { background-color: #f8f9fa; }
.results-table tr:hover { background-color: #e3f2fd; }
.download-section { background: #2ecc71; color: white; border-radius: 8px; padding: 20px; margin: 20px 0; text-align: center; }
.download-btn { background: #27ae60; color: white; border: none; border-radius: 5px; padding: 12px 25px; cursor: pointer; font-size: 16px; margin: 5px; text-decoration: none; display: inline-block; }
.download-btn:hover { background: #219a52; }
.back-btn { background: #34495e; color: white; text-decoration: none; padding: 12px 25px; border-radius: 8px; display: inline-block; margin-top: 20px; font-weight: bold; }
.back-btn:hover { background: #2c3e50; }
.highlight { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 8px; margin: 15px 0; }
.stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0; }
.stat-card { background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; border: 1px solid #dee2e6; }
.stat-value { font-size: 24px; font-weight: bold; color: #3498db; }
.stat-label { color: #6c757d; margin-top: 5px; }
</style>

<div class="container">
    <div class="header">
        <h1 style="color: #2c3e50; font-size: 2.5em; margin-bottom: 20px;">Finding Similar Product in E-commerce</h1>
        <h2 style="color: #34495e; margin-bottom: 10px;">Exercise 3: LSH Hyperparameter Evaluation</h2>
        <p>Comprehensive evaluation of LSH algorithm performance with different hyperparameters</p>
    </div>

    <div class="requirements">
        <h3>Exercise Requirements</h3>
        <ul>
            <li><strong>Evaluation Set:</strong> Top-100 products with highest number of similar products</li>
            <li><strong>Evaluation Metric:</strong> MAP@10 (Mean Average Precision)</li>
            <li><strong>Modes:</strong> PST (Product Similarity - Title) and PSD (Product Similarity - Description)</li>
            <li><strong>Hyperparameters to evaluate:</strong></li>
            <ul>
                <li>K-character shingles: K = 2, 3, 5, 7, 10</li>
                <li>Number of hash functions: 10, 20, 50, 100, 150</li>
                <li>LSH parameters: Various (b, r) combinations</li>
            </ul>
        </ul>
    </div>

    <div class="highlight">
        <strong>Dataset Status:</strong> 
        <span id="dataset-status">Loading dataset information...</span>
    </div>

    <div class="eval-section">
        <h3>Start Evaluation</h3>
        <p>Click the button below to start the comprehensive Exercise 3 evaluation. This will:</p>
        <ul>
            <li>Create evaluation set of top-100 products with most similar items</li>
            <li>Test all specified hyperparameter combinations</li>
            <li>Calculate MAP@10 for each configuration</li>
            <li>Generate tables and visualizations</li>
            <li>Create downloadable reports</li>
        </ul>
        
        <div style="text-align: center; margin-top: 25px;">
            <button onclick="runExercise3()" class="eval-btn" id="exercise3-btn">
                Run Exercise 3 Evaluation
            </button>
        </div>
        
        <div id="evaluation-status" style="display: none;"></div>
    </div>

    <div id="evaluation-results" style="display: none;">
        <div class="results-section">
            <h3>Evaluation Results</h3>
            
            <div id="eval-stats" class="stats-grid">
                <!-- Evaluation statistics will be populated here -->
            </div>
            
            <div id="results-tables">
                <!-- Results tables will be populated here -->
            </div>
            
            <div id="download-section" class="download-section" style="display: none;">
                <h4>Download Results</h4>
                <p>All tables, graphs, and reports have been generated and saved locally in the 'exercise3_results' directory.</p>
                <div id="download-links">
                    <!-- Download links will be populated here -->
                </div>
            </div>
        </div>
    </div>

    <a href="/" class="back-btn">← Back to Product List</a>
</div>

<script>
// Load dataset status when page loads
document.addEventListener('DOMContentLoaded', function() {
    fetch('/dataset_status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const statusDiv = document.getElementById('dataset-status');
                statusDiv.innerHTML = `
                    Total products: <strong>${data.total_products.toLocaleString()}</strong> | 
                    With titles (PST): <strong>${data.products_with_titles.toLocaleString()}</strong> | 
                    With descriptions (PSD): <strong>${data.products_with_descriptions.toLocaleString()}</strong> 
                    (${data.description_coverage}% coverage)
                    ${data.psd_mode_available ? 'Both PST and PSD modes available' : 'Only PST mode available'}
                `;
            }
        })
        .catch(error => {
            console.error('Error loading dataset status:', error);
        });
});

function runExercise3() {
    const button = document.getElementById('exercise3-btn');
    const statusDiv = document.getElementById('evaluation-status');
    const resultsDiv = document.getElementById('evaluation-results');
    
    // Show loading state
    button.textContent = 'Running Evaluation...';
    button.disabled = true;
    button.style.background = '#6c757d';
    
    statusDiv.style.display = 'block';
    statusDiv.innerHTML = '<div class="status-box">Starting Exercise 3 evaluation... This may take several minutes.</div>';
    
    // Make AJAX request
    fetch('/run_exercise3')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                statusDiv.innerHTML = '<div class="status-box success">Exercise 3 evaluation completed successfully!</div>';
                
                // Display results
                displayExercise3Results(data);
                resultsDiv.style.display = 'block';
                
                // Update button
                button.textContent = 'Evaluation Complete';
                button.style.background = '#27ae60';
            } else {
                statusDiv.innerHTML = '<div class="status-box error">Error: ' + data.error + '</div>';
                button.textContent = 'Run Exercise 3 Evaluation';
                button.style.background = '#e74c3c';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            statusDiv.innerHTML = '<div class="status-box error">Error running evaluation: ' + error + '</div>';
            button.textContent = 'Run Exercise 3 Evaluation';
            button.style.background = '#e74c3c';
        })
        .finally(() => {
            button.disabled = false;
        });
}

function displayExercise3Results(data) {
    // Display simple success message
    const statsDiv = document.getElementById('eval-stats');
    statsDiv.innerHTML = `
        <div class="stat-card" style="width: 100%; text-align: center; padding: 20px; background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; color: #155724;">
            <div class="stat-value" style="font-size: 2em; margin-bottom: 10px;">✓</div>
            <div class="stat-label" style="font-size: 1.2em; font-weight: bold;">Evaluation Completed Successfully!</div>
            <div style="margin-top: 10px; font-size: 0.9em;">
                All results have been saved to the <strong>${data.output_dir}</strong> folder
            </div>
        </div>
    `;
    
    // Clear results tables and show simple info
    const tablesDiv = document.getElementById('results-tables');
    tablesDiv.innerHTML = `
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; margin: 20px 0;">
            <h4>Files Generated</h4>
            <p>Number of files created: <strong>${data.files_count}</strong></p>
            ${data.sample_files.length > 0 ? 
                `<p>Sample files: ${data.sample_files.map(f => `<code>${f}</code>`).join(', ')}</p>` 
                : ''}
            <p style="margin-top: 15px; color: #666;">
                Check the <strong>${data.output_dir}</strong> folder in your project directory for all CSV files, visualizations, and analysis reports.
            </p>
        </div>
    `;
    
    // Show download section with simplified message
    const downloadSection = document.getElementById('download-section');
    downloadSection.style.display = 'block';
    
    const downloadLinks = document.getElementById('download-links');
    downloadLinks.innerHTML = `
        <p><strong>All results saved in '${data.output_dir}' directory</strong></p>
        <div style="text-align: center; padding: 20px;">
            <p style="color: #28a745; font-size: 1.1em; margin-bottom: 10px;">
                Exercise 3 evaluation is complete!
            </p>
            <p style="color: #666;">
                You can now find all CSV files, visualization plots, and analysis reports in your project folder.
            </p>
        </div>
    `;
}
</script>
'''

SIMILAR_TEMPLATE = '''
<style>
body { font-family: Arial, sans-serif; background: #f7f7f7; }
.container { display: flex; flex-direction: row; gap: 32px; margin: 32px auto; max-width: 1200px; background: #fff; box-shadow: 0 2px 8px #ccc; border-radius: 12px; padding: 32px; }
.left-col { flex: 1; border-right: 1px solid #eee; padding-right: 32px; }
.right-col { flex: 2; padding-left: 32px; }
.prod-img { max-width: 180px; max-height: 180px; border-radius: 8px; box-shadow: 0 1px 4px #bbb; margin-bottom: 16px; }
.prod-title { font-size: 1.3em; font-weight: bold; margin-bottom: 8px; color: #2a2a2a; }
.prod-desc { color: #444; margin-bottom: 12px; }
.prod-price { color: #007600; font-weight: bold; margin-bottom: 8px; }
.prod-link { color: #0056b3; text-decoration: none; font-size: 1em; }
.sim-list { display: flex; flex-wrap: wrap; gap: 24px; }
.sim-card { background: #fafafa; border-radius: 10px; box-shadow: 0 1px 4px #ddd; padding: 16px; width: 260px; display: flex; flex-direction: column; align-items: center; transition: box-shadow 0.2s; }
.sim-card:hover { box-shadow: 0 4px 16px #bbb; }
.sim-title { font-size: 1.1em; font-weight: bold; margin: 8px 0 4px 0; color: #333; text-align: center; }
.sim-desc { color: #555; font-size: 0.95em; margin-bottom: 8px; text-align: center; }
.sim-price { color: #007600; font-weight: bold; margin-bottom: 6px; }
.sim-link { color: #0056b3; text-decoration: none; font-size: 0.98em; }
.back-btn { margin-top: 32px; display: inline-block; background: #0056b3; color: #fff; padding: 10px 22px; border-radius: 6px; text-decoration: none; font-weight: bold; transition: background 0.2s; }
.back-btn:hover { background: #003d80; }
.eval-section { background: #f8f9fa; border-radius: 8px; padding: 20px; margin-top: 20px; border: 1px solid #e0e0e0; }
.eval-btn { background: #17a2b8; color: white; border: none; border-radius: 5px; padding: 8px 16px; cursor: pointer; margin-right: 10px; }
.eval-btn:hover { background: #138496; }
.eval-results { margin-top: 15px; }
.metric-card { background: white; border-radius: 6px; padding: 12px; margin: 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.metric-title { font-weight: bold; color: #333; margin-bottom: 8px; }
.metric-value { display: inline-block; margin-right: 15px; }
</style>
<div class="container">
    <div class="left-col">
        {% set img = query_prod.get('imageURLHighRes', []) or query_prod.get('imageURL', []) %}
        {% if img and img[0] %}
            <img src="{{ img[0] }}" alt="Product Image" class="prod-img">
        {% endif %}
        <div class="prod-title">{{ query_prod['title'] }}</div>
        <div class="prod-desc">{{ query_prod['description'] if query_prod['description'] else 'No description available.' }}</div>
        {% if query_prod.get('price') %}<div class="prod-price">Price: {{ query_prod['price'] }}</div>{% endif %}
        <a class="prod-link" href="https://www.amazon.com/dp/{{ query_prod['asin'] }}" target="_blank">View on Amazon</a>
        <div style="margin-top:24px; color:#888; font-size:0.95em;">Mode: <b>{{ mode }}</b></div>
    </div>
    <div class="right-col">
        <h3 style="margin-bottom:18px;">Similar Products</h3>
        <div class="sim-list">
        {% for prod in similar_prods %}
            <div class="sim-card">
                {% set img = prod.get('imageURLHighRes', []) or prod.get('imageURL', []) %}
                {% if img and img[0] %}
                    <img src="{{ img[0] }}" alt="Product Image" class="prod-img" style="max-width:120px;max-height:120px;">
                {% endif %}
                <div class="sim-title">{{ prod['title'] }}</div>
                <div class="sim-desc">{{ prod['description'][:80] if prod['description'] else 'No description.' }}{% if prod['description'] and prod['description']|length > 80 %}...{% endif %}</div>
                {% if prod.get('price') %}<div class="sim-price">Price: {{ prod['price'] }}</div>{% endif %}
                <a class="sim-link" href="https://www.amazon.com/dp/{{ prod['asin'] }}" target="_blank">View on Amazon</a>
            </div>
        {% endfor %}
        </div>
        {% if total_available < top_k %}
            <div style="color:#888; font-size:0.95em; margin-top:16px; padding:12px; background:#f9f9f9; border-radius:8px;">
                For this product, only <strong>{{ total_available }}</strong> similar products are available (requested k={{ top_k }}).
            </div>
        {% endif %}
        
        <!-- Evaluation Section -->
        <div class="eval-section">
            <h4>Evaluate Results</h4>
            <p style="color:#666; font-size:0.9em;">Compare algorithm results with ground truth similar products</p>
            <button onclick="evaluateResults({{ product_idx }}, '{{ mode }}', {{ top_k }})" class="eval-btn" id="eval-btn">
                Calculate Precision@k & ROUGE Scores
            </button>
            <div id="eval-results" style="display:none;">
                <div class="eval-results">
                    <div class="metric-card">
                        <div class="metric-title">Precision@k Results:</div>
                        <div id="precision-metrics"></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">ROUGE Scores (Title):</div>
                        <div id="rouge-title-metrics"></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">ROUGE Scores (Description):</div>
                        <div id="rouge-desc-metrics"></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">ROUGE Scores (Title+Description Combined):</div>
                        <div id="rouge-combined-metrics"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <a class="back-btn" href="/">&#8592; Back to Product List</a>
    </div>
</div>

<script>
function evaluateResults(productIdx, mode, topK) {
    const button = document.getElementById('eval-btn');
    const resultsDiv = document.getElementById('eval-results');
    
    // Show loading state
    button.textContent = 'Calculating...';
    button.disabled = true;
    button.style.background = '#6c757d';
    
    // Make AJAX request
    fetch('/evaluate_similar?product_idx=' + productIdx + '&mode=' + mode + '&top_k=' + topK)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.metrics) {
                // Show results
                resultsDiv.style.display = 'block';
                
                // Update precision metrics
                const precisionDiv = document.getElementById('precision-metrics');
                let precisionHtml = '';
                for (const [k, p] of Object.entries(data.metrics.precisions || {})) {
                    const value = p !== null ? p.toFixed(3) : 'N/A';
                    precisionHtml += '<span class="metric-value"><strong>' + k + ':</strong> ' + value + '</span>';
                }
                precisionDiv.innerHTML = precisionHtml;
                
                // Update ROUGE title metrics
                const rougeTitleDiv = document.getElementById('rouge-title-metrics');
                const titleScores = data.metrics.rouge_scores;
                let rougeTitleHtml = '';
                rougeTitleHtml += '<span class="metric-value"><strong>ROUGE-1:</strong> ' + (titleScores.rouge_1_title || 0).toFixed(3) + '</span>';
                rougeTitleHtml += '<span class="metric-value"><strong>ROUGE-2:</strong> ' + (titleScores.rouge_2_title || 0).toFixed(3) + '</span>';
                rougeTitleHtml += '<span class="metric-value"><strong>ROUGE-L:</strong> ' + (titleScores.rouge_l_title || 0).toFixed(3) + '</span>';
                rougeTitleDiv.innerHTML = rougeTitleHtml;
                
                // Update ROUGE description metrics
                const rougeDescDiv = document.getElementById('rouge-desc-metrics');
                let rougeDescHtml = '';
                if (data.metrics.rouge_scores.has_descriptions) {
                    rougeDescHtml += '<span class="metric-value"><strong>ROUGE-1:</strong> ' + (titleScores.rouge_1_desc || 0).toFixed(3) + '</span>';
                    rougeDescHtml += '<span class="metric-value"><strong>ROUGE-2:</strong> ' + (titleScores.rouge_2_desc || 0).toFixed(3) + '</span>';
                    rougeDescHtml += '<span class="metric-value"><strong>ROUGE-L:</strong> ' + (titleScores.rouge_l_desc || 0).toFixed(3) + '</span>';
                } else {
                    rougeDescHtml = '<span style="color: #999; font-style: italic;">No descriptions available in dataset</span>';
                }
                rougeDescDiv.innerHTML = rougeDescHtml;
                
                // Update ROUGE combined metrics
                const rougeCombinedDiv = document.getElementById('rouge-combined-metrics');
                let rougeCombinedHtml = '';
                rougeCombinedHtml += '<span class="metric-value"><strong>ROUGE-1:</strong> ' + (titleScores.rouge_1_combined || 0).toFixed(3) + '</span>';
                rougeCombinedHtml += '<span class="metric-value"><strong>ROUGE-2:</strong> ' + (titleScores.rouge_2_combined || 0).toFixed(3) + '</span>';
                rougeCombinedHtml += '<span class="metric-value"><strong>ROUGE-L:</strong> ' + (titleScores.rouge_l_combined || 0).toFixed(3) + '</span>';
                rougeCombinedDiv.innerHTML = rougeCombinedHtml;
                
                // Update button
                button.textContent = '✓ Evaluation Complete';
                button.style.background = '#28a745';
            } else {
                alert('Error evaluating results: ' + (data.error || 'No ground truth available'));
                button.textContent = 'Calculate Precision@k & ROUGE Scores';
                button.style.background = '#17a2b8';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error evaluating results');
            button.textContent = 'Calculate Precision@k & ROUGE Scores';
            button.style.background = '#17a2b8';
        })
        .finally(() => {
            button.disabled = false;
        });
}
</script>
'''

@app.route('/')
def home():
    page = int(request.args.get('page', 1))
    top_k = int(request.args.get('top_k', 5))
    
    # Sort products: highest similar_item count first, then those with none at the end
    def sim_count(prod):
        sim = prod.get('similar_item')
        if sim and isinstance(sim, list):
            return len(sim)
        elif sim and isinstance(sim, str):
            return 1 if sim else 0
        return 0
    
    sorted_products = sorted(list(enumerate(products)), key=lambda x: sim_count(x[1]), reverse=True)
    # Move products with zero similar_item to the end
    with_sim = [p for p in sorted_products if sim_count(p[1]) > 0]
    without_sim = [p for p in sorted_products if sim_count(p[1]) == 0]
    page_products = with_sim + without_sim
    start_idx = (page - 1) * PRODUCTS_PER_PAGE
    end_idx = min(start_idx + PRODUCTS_PER_PAGE, len(page_products))
    page_products = page_products[start_idx:end_idx]
    
    # Don't calculate precision by default - will be calculated individually
    products_with_precision = []
    for idx, prod in page_products:
        products_with_precision.append((idx, prod, None))
    
    return render_template_string(HOME_TEMPLATE, 
                                products=products_with_precision, 
                                page=page, 
                                end_idx=end_idx, 
                                total=len(products), 
                                top_k=top_k)

@app.route('/precision')
def precision():
    product_idx = int(request.args.get('product_idx', 0))
    top_k = int(request.args.get('top_k', 5))
    
    # Calculate precision and ROUGE for all modes for this specific product
    metrics = calculate_precisions_and_rouge_for_all_modes(product_idx)
    
    if metrics:
        # Extract just precisions for backward compatibility
        precisions = {}
        rouge_scores = {}
        for mode in metrics:
            precisions[mode] = metrics[mode]['precisions']
            rouge_scores[mode] = metrics[mode]['rouge_scores']
        
        # Return JSON response for AJAX
        return jsonify({
            'product_idx': product_idx,
            'precisions': precisions,
            'rouge_scores': rouge_scores,
            'success': True
        })
    else:
        return jsonify({
            'product_idx': product_idx,
            'precisions': None,
            'rouge_scores': None,
            'success': False,
            'error': 'No ground truth available'
        })

@app.route('/exercise3')
def exercise3_page():
    """Page for Exercise 3 evaluation"""
    return render_template_string(EXERCISE3_TEMPLATE)

@app.route('/dataset_status')
def dataset_status():
    """Get dataset status information"""
    try:
        pst_count = sum(1 for p in products if p.get('title', '').strip())
        psd_count = sum(1 for p in products if p.get('description', '').strip())
        
        return jsonify({
            'success': True,
            'total_products': len(products),
            'products_with_titles': pst_count,
            'products_with_descriptions': psd_count,
            'pst_mode_available': pst_count > 0,
            'psd_mode_available': psd_count > 0,
            'description_coverage': round((psd_count / len(products)) * 100, 1) if products else 0
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/run_exercise3')
def run_exercise3():
    """Run Exercise 3 evaluation and return simple success message"""
    try:
        from exercise3_evaluation import Exercise3Evaluator
        import os
        
        evaluator = Exercise3Evaluator()
        
        # Run the incremental evaluation (use fast_mode=False for full evaluation)
        results = evaluator.evaluate_exercise_3_with_incremental_output(fast_mode=False)
        
        # Check if results folder exists and has files
        output_dir = 'exercise3_results'
        files_created = []
        if os.path.exists(output_dir):
            files_created = [f for f in os.listdir(output_dir) if not f.startswith('.')]
        
        return jsonify({
            'success': True,
            'message': 'Exercise 3 evaluation completed successfully!',
            'output_dir': output_dir,
            'files_count': len(files_created),
            'sample_files': files_created[:5] if files_created else [],
            'eval_stats': {
                'total_products': evaluator.eval_stats.get('total_products', 'N/A'),
                'products_evaluated': len(evaluator.eval_set) if hasattr(evaluator, 'eval_set') else 'N/A'
            }
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Exercise 3 error: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_details': error_details
        })

@app.route('/evaluate_similar')
def evaluate_similar():
    product_idx = int(request.args.get('product_idx', 0))
    mode = request.args.get('mode', 'PST')
    top_k = int(request.args.get('top_k', 5))
    
    # Calculate metrics for this specific mode
    metrics = calculate_all_metrics(product_idx, mode, k=top_k)
    
    if metrics:
        return jsonify({
            'product_idx': product_idx,
            'mode': mode,
            'metrics': metrics,
            'success': True
        })
    else:
        return jsonify({
            'product_idx': product_idx,
            'mode': mode,
            'metrics': None,
            'success': False,
            'error': 'No ground truth available or calculation failed'
        })

@app.route('/similar')
def similar():
    product_idx = int(request.args.get('product_idx', 0))
    mode = request.args.get('mode', 'PST')
    top_k = int(request.args.get('top_k', 5))
    # Get all similar products first
    similar_idxs = query_similar_products(product_idx, mode=mode, top_k=100)  # Get more than needed
    similar_prods = [products[i] for i in similar_idxs if i != product_idx]
    
    # Now limit to top_k
    total_available = len(similar_prods)
    similar_prods = similar_prods[:top_k]  # Take only top k
    num_shown = len(similar_prods)
    
    query_prod = products[product_idx]
    return render_template_string(SIMILAR_TEMPLATE, 
                                query_prod=query_prod, 
                                similar_prods=similar_prods, 
                                mode=mode, 
                                num_shown=num_shown, 
                                top_k=top_k, 
                                total_available=total_available,
                                product_idx=product_idx)

if __name__ == '__main__':
        app.run(debug=True, port=5002)
