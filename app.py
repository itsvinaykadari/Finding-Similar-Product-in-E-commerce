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

def calculate_rouge_1(predicted_text, reference_text):
    """Calculate ROUGE-1 score (unigram overlap)"""
    if not predicted_text or not reference_text:
        return 0.0
    
    pred_words = predicted_text.lower().split()
    ref_words = reference_text.lower().split()
    
    if not ref_words:
        return 0.0
    
    pred_counter = Counter(pred_words)
    ref_counter = Counter(ref_words)
    
    overlap = sum((pred_counter & ref_counter).values())
    return overlap / len(ref_words)

def calculate_rouge_2(predicted_text, reference_text):
    """Calculate ROUGE-2 score (bigram overlap)"""
    if not predicted_text or not reference_text:
        return 0.0
    
    def get_bigrams(words):
        return [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    
    pred_words = predicted_text.lower().split()
    ref_words = reference_text.lower().split()
    
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
    """Calculate ROUGE-L score (Longest Common Subsequence)"""
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
    
    pred_words = predicted_text.lower().split()
    ref_words = reference_text.lower().split()
    
    if not ref_words:
        return 0.0
    
    lcs_len = lcs_length(pred_words, ref_words)
    return lcs_len / len(ref_words)

def calculate_rouge_scores(predicted_products, ground_truth_products):
    """Calculate ROUGE scores for titles, descriptions, and combined title+description"""
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
            'has_descriptions': False
        }
    
    # Concatenate titles and descriptions
    pred_titles = ' '.join([prod.get('title', '') for prod in predicted_products])
    ref_titles = ' '.join([prod.get('title', '') for prod in ground_truth_products])
    
    pred_descs = ' '.join([prod.get('description', '') for prod in predicted_products])
    ref_descs = ' '.join([prod.get('description', '') for prod in ground_truth_products])
    
    # Combined title + description
    pred_combined = ' '.join([
        (prod.get('title', '') + ' ' + prod.get('description', '')).strip() 
        for prod in predicted_products
    ])
    ref_combined = ' '.join([
        (prod.get('title', '') + ' ' + prod.get('description', '')).strip() 
        for prod in ground_truth_products
    ])
    
    # Check if descriptions are actually available
    has_descriptions = bool(pred_descs.strip() and ref_descs.strip())
    
    return {
        'rouge_1_title': calculate_rouge_1(pred_titles, ref_titles),
        'rouge_2_title': calculate_rouge_2(pred_titles, ref_titles),
        'rouge_l_title': calculate_rouge_l(pred_titles, ref_titles),
        'rouge_1_desc': calculate_rouge_1(pred_descs, ref_descs) if has_descriptions else 0.0,
        'rouge_2_desc': calculate_rouge_2(pred_descs, ref_descs) if has_descriptions else 0.0,
        'rouge_l_desc': calculate_rouge_l(pred_descs, ref_descs) if has_descriptions else 0.0,
        'rouge_1_combined': calculate_rouge_1(pred_combined, ref_combined),
        'rouge_2_combined': calculate_rouge_2(pred_combined, ref_combined),
        'rouge_l_combined': calculate_rouge_l(pred_combined, ref_combined),
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
        # Get LSH results
        lsh_results = query_similar_products(product_idx, mode=mode, top_k=k)
        lsh_results = [idx for idx in lsh_results if idx != product_idx][:k]
        
        # Get ground truth
        ground_truth_indices = get_ground_truth_products(product_idx)
        
        if not ground_truth_indices:
            return None
        
        # Calculate precision@k for different k values
        precisions = {}
        for pk in [1, 3, 5, 10]:
            if pk <= len(lsh_results):
                relevant_count = len(set(lsh_results[:pk]) & set(ground_truth_indices))
                precisions[f'p@{pk}'] = relevant_count / pk if pk > 0 else 0.0
            else:
                precisions[f'p@{pk}'] = None
        
        # Get product objects for ROUGE calculation
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
    <h2 style="margin-bottom:24px;">Product List (Page {{ page }})</h2>
    <div style="margin-bottom: 20px;">
        <a href="/exercise3" style="background: #e74c3c; color: white; text-decoration: none; padding: 12px 25px; border-radius: 8px; font-weight: bold; margin-right: 15px;">
            📊 Exercise 3: LSH Evaluation
        </a>
        <a href="/evaluation" style="background: #17a2b8; color: white; text-decoration: none; padding: 10px 20px; border-radius: 5px; font-weight: bold;">
            🔧 Advanced Hyperparameter Testing
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

EVALUATION_TEMPLATE = '''
<style>
body { font-family: Arial, sans-serif; background: #f7f7f7; }
.container { max-width: 1200px; margin: 32px auto; background: #fff; box-shadow: 0 2px 8px #ccc; border-radius: 12px; padding: 32px; }
.eval-section { background: #f8f9fa; border-radius: 8px; padding: 20px; margin: 20px 0; border: 1px solid #e0e0e0; }
.eval-btn { background: #28a745; color: white; border: none; border-radius: 5px; padding: 12px 24px; cursor: pointer; font-size: 16px; margin: 10px 5px; }
.eval-btn:hover { background: #218838; }
.eval-btn.secondary { background: #17a2b8; }
.eval-btn.secondary:hover { background: #138496; }
.eval-btn:disabled { background: #6c757d; cursor: not-allowed; }
.results-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
.results-table th, .results-table td { border: 1px solid #ddd; padding: 8px; text-align: center; }
.results-table th { background-color: #f2f2f2; }
.back-btn { background: #007bff; color: white; text-decoration: none; padding: 10px 20px; border-radius: 5px; display: inline-block; margin-top: 20px; }
.back-btn:hover { background: #0056b3; }
.warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0; }
</style>

<div class="container">
    <h2>� Exercise 3: LSH Hyperparameter Evaluation</h2>
    <p>Comprehensive LSH algorithm evaluation with grid search across all parameter combinations.</p>
    
    <div class="eval-section">
        <h3>✅ Features Available</h3>
        <p>Exercise 3 provides comprehensive LSH evaluation with:</p>
        <ul>
            <li>✅ Top-100 products evaluation set</li>
            <li>✅ MAP@10 calculations</li>
            <li>✅ K-character shingles testing (2, 3, 5, 7, 10)</li>
            <li>✅ Hash functions evaluation (10, 20, 50, 100, 150)</li>
            <li>✅ LSH parameters analysis (various b, r combinations)</li>
            <li>✅ Grid search across all parameter combinations</li>
            <li>✅ Advanced visualization and statistical analysis</li>
            <li>✅ Downloadable tables and graphs</li>
            <li>✅ Export functionality for research reports</li>
        </ul>
        
        <a href="/exercise3" style="background: #e74c3c; color: white; text-decoration: none; padding: 12px 25px; border-radius: 8px; font-weight: bold; display: inline-block; margin-top: 10px;">
            🏁 Start Exercise 3 Evaluation
        </a>
    </div>
    
    <a href="/" class="back-btn">← Back to Product List</a>
</div>
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
        <h1>📊 Exercise 3: LSH Hyperparameter Evaluation</h1>
        <p>Comprehensive evaluation of LSH algorithm performance with different hyperparameters</p>
    </div>

    <div class="requirements">
        <h3>🎯 Exercise Requirements</h3>
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
        <strong>📝 Note:</strong> This evaluation will generate comprehensive tables and graphs that will be automatically downloaded to your local machine. The evaluation focuses on PST and PSD modes separately (no PSTD as per requirements).
    </div>

    <div class="eval-section">
        <h3>🚀 Start Evaluation</h3>
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
                🏁 Run Exercise 3 Evaluation
            </button>
        </div>
        
        <div id="evaluation-status" style="display: none;"></div>
    </div>

    <div id="evaluation-results" style="display: none;">
        <div class="results-section">
            <h3>📈 Evaluation Results</h3>
            
            <div id="eval-stats" class="stats-grid">
                <!-- Evaluation statistics will be populated here -->
            </div>
            
            <div id="results-tables">
                <!-- Results tables will be populated here -->
            </div>
            
            <div id="download-section" class="download-section" style="display: none;">
                <h4>📥 Download Results</h4>
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
function runExercise3() {
    const button = document.getElementById('exercise3-btn');
    const statusDiv = document.getElementById('evaluation-status');
    const resultsDiv = document.getElementById('evaluation-results');
    
    // Show loading state
    button.textContent = '⏳ Running Evaluation...';
    button.disabled = true;
    button.style.background = '#6c757d';
    
    statusDiv.style.display = 'block';
    statusDiv.innerHTML = '<div class="status-box">🔄 Starting Exercise 3 evaluation... This may take several minutes.</div>';
    
    // Make AJAX request
    fetch('/run_exercise3')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                statusDiv.innerHTML = '<div class="status-box success">✅ Exercise 3 evaluation completed successfully!</div>';
                
                // Display results
                displayExercise3Results(data);
                resultsDiv.style.display = 'block';
                
                // Update button
                button.textContent = '✅ Evaluation Complete';
                button.style.background = '#27ae60';
            } else {
                statusDiv.innerHTML = '<div class="status-box error">❌ Error: ' + data.error + '</div>';
                button.textContent = '🏁 Run Exercise 3 Evaluation';
                button.style.background = '#e74c3c';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            statusDiv.innerHTML = '<div class="status-box error">❌ Error running evaluation: ' + error + '</div>';
            button.textContent = '🏁 Run Exercise 3 Evaluation';
            button.style.background = '#e74c3c';
        })
        .finally(() => {
            button.disabled = false;
        });
}

function displayExercise3Results(data) {
    // Display evaluation statistics
    const statsDiv = document.getElementById('eval-stats');
    statsDiv.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${data.eval_stats.total_products}</div>
            <div class="stat-label">Products Evaluated</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${data.eval_stats.max_similar_items}</div>
            <div class="stat-label">Max Similar Items</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${data.eval_stats.min_similar_items}</div>
            <div class="stat-label">Min Similar Items</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${data.eval_stats.avg_similar_items.toFixed(1)}</div>
            <div class="stat-label">Avg Similar Items</div>
        </div>
    `;
    
    // Display results tables
    const tablesDiv = document.getElementById('results-tables');
    let tablesHtml = '';
    
    const modes = ['PST', 'PSD'];
    const categories = ['shingle_k', 'num_hashes', 'lsh_params'];
    const categoryNames = ['K-character Shingles', 'Number of Hash Functions', 'LSH Parameters (b, r)'];
    
    for (const mode of modes) {
        tablesHtml += `<h4>📋 ${mode} Mode Results</h4>`;
        
        for (let i = 0; i < categories.length; i++) {
            const category = categories[i];
            const categoryName = categoryNames[i];
            
            tablesHtml += `<h5>${categoryName}</h5>`;
            tablesHtml += '<table class="results-table">';
            
            if (category === 'lsh_params') {
                tablesHtml += '<thead><tr><th>b</th><th>r</th><th>MAP@10</th></tr></thead>';
                tablesHtml += '<tbody>';
                
                // Sort by MAP@10 score descending
                const sortedEntries = Object.entries(data.results[mode][category])
                    .sort((a, b) => b[1] - a[1]);
                
                for (const [params, score] of sortedEntries) {
                    const [b, r] = params.split(',').map(x => x.trim());
                    tablesHtml += `<tr><td>${b}</td><td>${r}</td><td>${score.toFixed(4)}</td></tr>`;
                }
            } else {
                const header = category === 'shingle_k' ? 'K' : 'Hash Functions';
                tablesHtml += `<thead><tr><th>${header}</th><th>MAP@10</th></tr></thead>`;
                tablesHtml += '<tbody>';
                
                // Sort by MAP@10 score descending
                const sortedEntries = Object.entries(data.results[mode][category])
                    .sort((a, b) => b[1] - a[1]);
                
                for (const [param, score] of sortedEntries) {
                    tablesHtml += `<tr><td>${param}</td><td>${score.toFixed(4)}</td></tr>`;
                }
            }
            
            tablesHtml += '</tbody></table>';
        }
    }
    
    tablesDiv.innerHTML = tablesHtml;
    
    // Show download section
    const downloadSection = document.getElementById('download-section');
    downloadSection.style.display = 'block';
    
    const downloadLinks = document.getElementById('download-links');
    downloadLinks.innerHTML = `
        <p><strong>Generated Files in '${data.output_dir}' directory:</strong></p>
        <div style="text-align: left; max-width: 600px; margin: 0 auto;">
            <ul style="color: white;">
                <li>📊 Comprehensive visualization plots</li>
                <li>📈 Individual mode analysis plots</li>
                <li>📋 CSV tables for each hyperparameter</li>
                <li>📄 Detailed summary report</li>
            </ul>
        </div>
        <p><em>All files have been saved to your local directory and are ready for use in your report!</em></p>
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

@app.route('/evaluation')
def evaluation_page():
    """Page for advanced hyperparameter evaluation"""
    return render_template_string(EVALUATION_TEMPLATE)

@app.route('/exercise3')
def exercise3_page():
    """Page for Exercise 3 evaluation"""
    return render_template_string(EXERCISE3_TEMPLATE)

@app.route('/run_exercise3')
def run_exercise3():
    """Run Exercise 3 evaluation and return results"""
    try:
        from exercise3_evaluation import Exercise3Evaluator
        evaluator = Exercise3Evaluator()
        
        # Run the evaluation
        results = evaluator.evaluate_exercise_3()
        
        # Generate tables and graphs
        output_dir = evaluator.generate_tables_and_graphs(results)
        
        return jsonify({
            'success': True,
            'results': results,
            'output_dir': output_dir,
            'eval_stats': evaluator.eval_stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
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
