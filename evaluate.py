import json
from query import query_similar_products
from collections import Counter
import numpy as np
from typing import List, Set

def precision_at_k(pred: List[str], truth: Set[str], k: int = 10) -> float:
    pred_k = pred[:k]
    if not truth:
        return 0.0
    return len([x for x in pred_k if x in truth]) / min(k, len(truth))

def average_precision(pred: List[str], truth: Set[str], k: int = 10) -> float:
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(pred[:k]):
        if p in truth:
            num_hits += 1
            score += num_hits / (i + 1)
    if not truth:
        return 0.0
    return score / min(len(truth), k)

def rouge_l(pred: str, truth: str) -> float:
    # Simple ROUGE-L implementation (Longest Common Subsequence)
    def lcs(X, Y):
        m, n = len(X), len(Y)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                if X[i] == Y[j]:
                    dp[i+1][j+1] = dp[i][j]+1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
        return dp[m][n]
    lcs_len = lcs(pred, truth)
    if len(truth) == 0:
        return 0.0
    return lcs_len / len(truth)

def get_top100_products(products):
    # Products with the most similar items ("similar_item" field)
    scored = [(i, len(set(prod.get('similar_item', [])))) for i, prod in enumerate(products)]
    scored = [x for x in scored if x[1] > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored[:100]]

def main():
    with open("processed_appliances.json", "r", encoding="utf-8") as f:
        products = json.load(f)

    top100 = get_top100_products(products)
    print(f"Top-100 products selected. Max similar items: {max(len(products[i]['similar_item']) for i in top100)}, Min: {min(len(products[i]['similar_item']) for i in top100)}")

    results = {"PST": [], "PSD": []}
    rouge_results = {"PST": [], "PSD": []}

    for mode in ["PST", "PSD"]:
        print(f"\nEvaluating mode: {mode}")
        for idx in top100:
            truth = set(products[idx].get('similar_item', []))
            pred_idxs = query_similar_products(idx, mode=mode, top_k=10)
            pred_asins = [products[i]['asin'] for i in pred_idxs if i != idx]
            prec = precision_at_k(pred_asins, truth, k=10)
            ap = average_precision(pred_asins, truth, k=10)
            results[mode].append((prec, ap))
            # ROUGE-L between concatenated titles of predicted and ground truth
            pred_titles = ' '.join([products[i]['title'] for i in pred_idxs if i != idx])
            truth_titles = ' '.join([prod['title'] for prod in products if prod['asin'] in truth])
            rouge = rouge_l(pred_titles, truth_titles)
            rouge_results[mode].append(rouge)
        mean_prec = np.mean([x[0] for x in results[mode]])
        mean_ap = np.mean([x[1] for x in results[mode]])
        mean_rouge = np.mean(rouge_results[mode])
        print(f"Mode: {mode} | MAP@10: {mean_ap:.4f} | Mean Precision@10: {mean_prec:.4f} | Mean ROUGE-L: {mean_rouge:.4f}")

if __name__ == "__main__":
    main()