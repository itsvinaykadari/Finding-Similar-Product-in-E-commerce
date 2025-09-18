# app.py
# Flask UI for interacting with the dataset and LSH index.

from flask import Flask, render_template_string, request
import json
from query import query_similar_products

app = Flask(__name__)

# Load products once at startup
with open("processed_appliances.json", "r", encoding="utf-8") as f:
        products = json.load(f)

PRODUCTS_PER_PAGE = 20


HOME_TEMPLATE = '''
<style>
body { font-family: Arial, sans-serif; background: #f7f7f7; }
.main-container { max-width: 1200px; margin: 32px auto; background: #fff; box-shadow: 0 2px 8px #ccc; border-radius: 12px; padding: 32px; }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 32px; }
.prod-card { background: #fafafa; border-radius: 10px; box-shadow: 0 1px 4px #ddd; padding: 18px; display: flex; flex-direction: column; align-items: center; min-height: 320px; }
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
</style>
<div class="main-container">
    <h2 style="margin-bottom:24px;">Product List (Page {{ page }})</h2>
    <form class="topk-form" method="get" action="/">
        <label for="top_k">Show top-k similar products (k): </label>
        <input type="number" id="top_k" name="top_k" min="1" max="20" value="{{ top_k }}" style="width:60px;">
        <button type="submit" class="sim-btn">Apply</button>
    </form>
    <div class="grid">
    {% for idx, prod in products %}
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
            {% endif %}
        </div>
    {% endfor %}
    </div>
    <div class="pagination">
        {% if page > 1 %}<a href="/?page={{ page-1 }}&top_k={{ top_k }}">&#8592; Previous</a>{% endif %}
        {% if end_idx < total %}<a href="/?page={{ page+1 }}&top_k={{ top_k }}">Next &#8594;</a>{% endif %}
    </div>
</div>
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
        <a class="back-btn" href="/">&#8592; Back to Product List</a>
    </div>
</div>
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
    return render_template_string(HOME_TEMPLATE, products=page_products, page=page, end_idx=end_idx, total=len(products), top_k=top_k)

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
    return render_template_string(SIMILAR_TEMPLATE, query_prod=query_prod, similar_prods=similar_prods, mode=mode, num_shown=num_shown, top_k=top_k, total_available=total_available)

if __name__ == '__main__':
        app.run(debug=True)
