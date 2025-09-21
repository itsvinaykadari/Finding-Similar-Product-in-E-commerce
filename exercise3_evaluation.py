import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datasketch import MinHashLSH, MinHash
from index_lsh import get_shingles, create_minhash
from collections import defaultdict
import os
from multiprocessing import Pool, cpu_count
import functools

class Exercise3Evaluator:
    def __init__(self, products_file="processed_appliances.json", sample_size=None):
        """Initialize evaluator for Exercise 3"""
        with open(products_file, "r", encoding="utf-8") as f:
            self.products = json.load(f)
        
        # Optional: Use a sample for faster testing
        if sample_size and sample_size < len(self.products):
            print(f"Using sample of {sample_size} products for faster evaluation")
            import random
            random.seed(42)  # For reproducible results
            self.products = random.sample(self.products, sample_size)
        
        # Create ASIN to index mapping
        self.asin_to_idx = {prod['asin']: i for i, prod in enumerate(self.products)}
        
        # Add a cache for pre-computed MinHashes
        self.minhash_cache = {}
        
        # Create evaluation set: top-100 products with most similar items
        self.eval_set, self.eval_stats = self._create_evaluation_set()
        print(f"Exercise 3 Evaluation set created with {len(self.eval_set)} products")
        
    def _create_evaluation_set(self):
        """Create evaluation set of top-100 products with most similar items"""
        products_with_sim_count = []
        
        for i, product in enumerate(self.products):
            similar_items = product.get('similar_item', [])
            sim_count = self._count_similar_items(similar_items)
            if sim_count > 0:
                products_with_sim_count.append((i, sim_count))
        
        # Sort by similarity count (descending) and take top 100
        products_with_sim_count.sort(key=lambda x: x[1], reverse=True)
        eval_set = [idx for idx, _ in products_with_sim_count[:100]]
        
        # Calculate statistics
        sim_counts = [count for _, count in products_with_sim_count[:100]]
        stats = {
            'max_similar_items': max(sim_counts),
            'min_similar_items': min(sim_counts),
            'avg_similar_items': np.mean(sim_counts),
            'total_products': len(eval_set)
        }
        
        print(f"Evaluation Set Statistics:")
        print(f"Max similar items: {stats['max_similar_items']}")
        print(f"Min similar items: {stats['min_similar_items']}")
        print(f"Average similar items: {stats['avg_similar_items']:.2f}")
        
        return eval_set, stats
    
    def _count_similar_items(self, similar_items):
        """Count number of valid similar items"""
        if not similar_items:
            return 0
            
        if isinstance(similar_items, str) and similar_items.strip():
            # Extract ASINs from HTML
            asin_pattern = r'/dp/([A-Z0-9]{10})'
            matches = re.findall(asin_pattern, similar_items)
            valid_asins = [asin for asin in set(matches) if asin in self.asin_to_idx]
            return len(valid_asins)
        elif isinstance(similar_items, list):
            count = 0
            for item in similar_items:
                if isinstance(item, str) and item.strip():
                    asin_pattern = r'/dp/([A-Z0-9]{10})'
                    matches = re.findall(asin_pattern, item)
                    valid_asins = [asin for asin in set(matches) if asin in self.asin_to_idx]
                    count += len(valid_asins)
            return count
        
        return 0
    
    def get_ground_truth(self, product_idx):
        """Get ground truth similar products for a product"""
        product = self.products[product_idx]
        similar_items = product.get('similar_item', [])
        
        asins = []
        if isinstance(similar_items, str) and similar_items.strip():
            asin_pattern = r'/dp/([A-Z0-9]{10})'
            matches = re.findall(asin_pattern, similar_items)
            asins = list(set(matches))
        elif isinstance(similar_items, list):
            for item in similar_items:
                if isinstance(item, str) and item.strip():
                    asin_pattern = r'/dp/([A-Z0-9]{10})'
                    matches = re.findall(asin_pattern, item)
                    asins.extend(matches)
            asins = list(set(asins))
        
        # Convert ASINs to indices
        ground_truth_indices = []
        for asin in asins:
            if asin in self.asin_to_idx:
                ground_truth_indices.append(self.asin_to_idx[asin])
        
        return ground_truth_indices
    
    def calculate_precision_at_k(self, predicted, ground_truth, k):
        """Calculate precision@k for a single query"""
        if not ground_truth or not predicted:
            return 0.0
        
        predicted_k = predicted[:k]
        relevant_count = len(set(predicted_k) & set(ground_truth))
        return relevant_count / k if k > 0 else 0.0
    
    def calculate_average_precision_at_k(self, predicted, ground_truth, k=10):
        """Calculate Average Precision at k for a single query"""
        if not ground_truth:
            return 0.0
        
        predicted_k = predicted[:k]
        relevant_count = 0
        precision_sum = 0.0
        
        for i, pred_idx in enumerate(predicted_k):
            if pred_idx in ground_truth:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / min(len(ground_truth), k)
    
    def _get_or_create_minhashes(self, mode, shingle_k, num_hashes):
        """
        Computes MinHashes for all products for a given configuration,
        or retrieves them from cache if already computed.
        """
        cache_key = (mode, shingle_k, num_hashes)
        if cache_key in self.minhash_cache:
            print(f"Using cached MinHashes for {cache_key}...")
            return self.minhash_cache[cache_key]

        print(f"Creating and caching MinHashes for {cache_key}...")
        minhashes = {}
        valid_products = 0
        
        for i, product in enumerate(self.products):
            if mode == 'PST':
                text = product.get('title', '')
            elif mode == 'PSD':
                text = product.get('description', '')
            else:
                text = ""

            if text and text.strip():  # Check for non-empty text
                shingles = get_shingles(text, shingle_k)
                if shingles:
                    minhashes[i] = create_minhash(shingles, num_hashes)
                    valid_products += 1
        
        print(f"Created MinHashes for {valid_products} products with valid {mode} text")
        self.minhash_cache[cache_key] = minhashes
        return minhashes
    
    def calculate_map_at_k_with_params(self, mode, shingle_k, num_hashes, b, r, k=10):
        """
        Calculate MAP@k with specific hyperparameters using pre-computed MinHashes.
        This version is much faster and removes the exhaustive fallback.
        """
        try:
            # 1. Get pre-computed MinHashes for this configuration
            all_minhashes = self._get_or_create_minhashes(mode, shingle_k, num_hashes)

            # 2. Create and populate the LSH index (this is now much faster)
            # Use a more lenient threshold calculation to get better results
            threshold = (1.0 / b) ** (1.0 / r)
            # Make threshold more lenient (lower values = more candidates)
            threshold = max(0.1, min(0.8, threshold))  # Keep threshold in reasonable range
            
            lsh = MinHashLSH(threshold=threshold, num_perm=num_hashes)
            
            print(f"Building LSH index with threshold={threshold:.3f} (b={b}, r={r})...")
            with lsh.insertion_session() as session:
                for i, minhash in all_minhashes.items():
                    session.insert(str(i), minhash)

            # 3. Calculate MAP@k for the evaluation set
            ap_scores = []
            for product_idx in self.eval_set:
                # Get the query minhash (already computed)
                query_minhash = all_minhashes.get(product_idx)
                if query_minhash is None:
                    continue

                # Get similar products from LSH
                result_keys = lsh.query(query_minhash)
                similar_indices = [int(key) for key in result_keys if int(key) != product_idx]
                
                # If LSH returns too few results, lower the threshold temporarily
                if len(similar_indices) < 5:  # Need at least a few candidates
                    # Create a more lenient LSH for this query
                    lenient_threshold = max(0.05, threshold * 0.5)
                    lenient_lsh = MinHashLSH(threshold=lenient_threshold, num_perm=num_hashes)
                    with lenient_lsh.insertion_session() as session:
                        for i, minhash in all_minhashes.items():
                            session.insert(str(i), minhash)
                    
                    result_keys = lenient_lsh.query(query_minhash)
                    similar_indices = [int(key) for key in result_keys if int(key) != product_idx]
                
                # Take only top k results
                similar_indices = similar_indices[:k]
                
                # Get ground truth
                ground_truth = self.get_ground_truth(product_idx)
                
                if ground_truth:
                    ap = self.calculate_average_precision_at_k(similar_indices, ground_truth, k=k)
                    ap_scores.append(ap)
            
            map_score = np.mean(ap_scores) if ap_scores else 0.0
            return map_score, ap_scores

        except Exception as e:
            print(f"Error in calculate_map_at_k_with_params: {e}")
            return 0.0, []
    
    def evaluate_exercise_3(self, fast_mode=False):
        """
        Evaluate Exercise 3 requirements with proper grid search
        fast_mode: If True, uses reduced parameter grid for faster testing
        """
        print("Starting Exercise 3 Evaluation with Grid Search...")
        if fast_mode:
            print("⚡ Fast mode enabled - using reduced parameter grid")
        
        results = {
            'PST': {'shingle_k': {}, 'num_hashes': {}, 'lsh_params': {}},
            'PSD': {'shingle_k': {}, 'num_hashes': {}, 'lsh_params': {}}
        }
        
        # Define hyperparameter ranges
        if fast_mode:
            shingle_k_values = [3, 5]  # Reduced from [2, 3, 5, 7, 10]
            num_hashes_values = [50, 100]  # Reduced from [10, 20, 50, 100, 150]
            lsh_configs = [
                (10, 10),  # 10*10 = 100
                (20, 5),   # 20*5 = 100
            ]
        else:
            shingle_k_values = [2, 3, 5, 7, 10]
            num_hashes_values = [10, 20, 50, 100, 150]
            lsh_configs = [
                (4, 25),   # 4*25 = 100
                (5, 20),   # 5*20 = 100
                (10, 10),  # 10*10 = 100
                (20, 5),   # 20*5 = 100
                (25, 4),   # 25*4 = 100
            ]
        
        # Define optimal baseline parameters for each test
        baseline_params = {
            'shingle_k': 3,
            'num_hashes': 100,
            'b': 20,
            'r': 5
        }
        
        # Modes to evaluate - check what data is available
        modes = []
        
        # Test PST mode
        pst_count = sum(1 for p in self.products if p.get('title', '').strip())
        if pst_count > 0:
            modes.append('PST')
            print(f"PST mode enabled: {pst_count} products with titles")
        
        # Test PSD mode  
        psd_count = sum(1 for p in self.products if p.get('description', '').strip())
        if psd_count > 0:
            modes.append('PSD')
            print(f"PSD mode enabled: {psd_count} products with descriptions")
        else:
            print(f"PSD mode skipped: No products have descriptions in this dataset")
        
        if not modes:
            print("ERROR: No valid modes available for evaluation!")
            return {}
        
        for mode in modes:
            print(f"\n=== Evaluating Mode: {mode} ===")
            
            # 1. Test K-character shingles (keeping other params fixed)
            print(f"\n1. Testing K-character shingles for {mode}...")
            for k in shingle_k_values:
                print(f"Testing shingle K={k} (num_hashes={baseline_params['num_hashes']}, b={baseline_params['b']}, r={baseline_params['r']})...")
                map_score, _ = self.calculate_map_at_k_with_params(
                    mode=mode, 
                    shingle_k=k, 
                    num_hashes=baseline_params['num_hashes'], 
                    b=baseline_params['b'], 
                    r=baseline_params['r'],
                    k=10
                )
                results[mode]['shingle_k'][k] = map_score
                print(f"K={k}: MAP@10 = {map_score:.4f}")
            
            # 2. Test number of hash functions (keeping other params fixed)
            print(f"\n2. Testing number of hash functions for {mode}...")
            for num_hashes in num_hashes_values:
                # Adjust b and r to match num_hashes while keeping the ratio similar
                if num_hashes <= 20:
                    b, r = min(num_hashes, 5), max(1, num_hashes // 5)
                elif num_hashes <= 50:
                    b, r = 10, num_hashes // 10
                elif num_hashes <= 100:
                    b, r = 20, num_hashes // 20
                else:  # 150
                    b, r = 30, num_hashes // 30
                
                print(f"Testing {num_hashes} hash functions (shingle_k={baseline_params['shingle_k']}, b={b}, r={r})...")
                map_score, _ = self.calculate_map_at_k_with_params(
                    mode=mode, 
                    shingle_k=baseline_params['shingle_k'], 
                    num_hashes=num_hashes, 
                    b=b, 
                    r=r,
                    k=10
                )
                results[mode]['num_hashes'][num_hashes] = map_score
                print(f"Hashes={num_hashes}: MAP@10 = {map_score:.4f}")
            
            # 3. Test LSH parameters (b, r) with fixed num_hashes=100
            print(f"\n3. Testing LSH parameters for {mode}...")
            for b, r in lsh_configs:
                print(f"Testing b={b}, r={r} (shingle_k={baseline_params['shingle_k']}, num_hashes=100)...")
                map_score, _ = self.calculate_map_at_k_with_params(
                    mode=mode, 
                    shingle_k=baseline_params['shingle_k'], 
                    num_hashes=100, 
                    b=b, 
                    r=r,
                    k=10
                )
                results[mode]['lsh_params'][(b, r)] = map_score
                print(f"b={b}, r={r}: MAP@10 = {map_score:.4f}")
        
        return results
    
    def evaluate_exercise_3_with_incremental_output(self, fast_mode=False):
        """
        Evaluate Exercise 3 with incremental output and progress tracking
        """
        print("Starting Exercise 3 Evaluation with Incremental Output...")
        if fast_mode:
            print("⚡ Fast mode enabled - using reduced parameter grid")
        
        results = {
            'PST': {'shingle_k': {}, 'num_hashes': {}, 'lsh_params': {}},
            'PSD': {'shingle_k': {}, 'num_hashes': {}, 'lsh_params': {}}
        }
        
        # Create output directory for incremental results
        output_dir = "exercise3_incremental_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Define hyperparameter ranges
        if fast_mode:
            shingle_k_values = [3, 5]
            num_hashes_values = [50, 100]
            lsh_configs = [(10, 10), (20, 5)]
        else:
            shingle_k_values = [2, 3, 5, 7, 10]
            num_hashes_values = [10, 20, 50, 100, 150]
            lsh_configs = [(4, 25), (5, 20), (10, 10), (20, 5), (25, 4)]
        
        baseline_params = {'shingle_k': 3, 'num_hashes': 100, 'b': 20, 'r': 5}
        
        # Check available modes
        modes = []
        pst_count = sum(1 for p in self.products if p.get('title', '').strip())
        psd_count = sum(1 for p in self.products if p.get('description', '').strip())
        
        if pst_count > 0:
            modes.append('PST')
            print(f"✅ PST mode available: {pst_count} products with titles")
        if psd_count > 0:
            modes.append('PSD')
            print(f"✅ PSD mode available: {psd_count} products with descriptions")
        else:
            print("❌ PSD mode unavailable: No products with descriptions")
        
        if not modes:
            print("ERROR: No valid modes available!")
            return {}
        
        # Calculate total tests for progress tracking
        total_tests = len(modes) * (len(shingle_k_values) + len(num_hashes_values) + len(lsh_configs))
        current_test = 0
        
        for mode in modes:
            print(f"\n{'='*60}")
            print(f"EVALUATING MODE: {mode}")
            print(f"{'='*60}")
            
            # 1. Test K-character shingles
            print(f"\n📊 TESTING K-CHARACTER SHINGLES ({mode})")
            print("-" * 50)
            
            for k in shingle_k_values:
                current_test += 1
                print(f"[{current_test}/{total_tests}] Testing shingle K={k}...")
                
                map_score, _ = self.calculate_map_at_k_with_params(
                    mode=mode, shingle_k=k, 
                    num_hashes=baseline_params['num_hashes'], 
                    b=baseline_params['b'], r=baseline_params['r'], k=10
                )
                
                results[mode]['shingle_k'][k] = map_score
                print(f"✅ K={k}: MAP@10 = {map_score:.4f}")
                
                # Save incremental results
                self._save_incremental_results(results, output_dir, f"{mode}_shingle_k_progress")
            
            # Generate intermediate table for shingle results
            self._generate_intermediate_table(results[mode]['shingle_k'], 
                                            f"{output_dir}/{mode}_shingle_k_results.csv",
                                            "K", "Shingle K Results")
            
            # 2. Test number of hash functions
            print(f"\n🔢 TESTING NUMBER OF HASH FUNCTIONS ({mode})")
            print("-" * 50)
            
            for num_hashes in num_hashes_values:
                current_test += 1
                
                # Adjust b and r for different hash counts
                if num_hashes <= 20:
                    b, r = min(num_hashes, 5), max(1, num_hashes // 5)
                elif num_hashes <= 50:
                    b, r = 10, num_hashes // 10
                elif num_hashes <= 100:
                    b, r = 20, num_hashes // 20
                else:
                    b, r = 30, num_hashes // 30
                
                print(f"[{current_test}/{total_tests}] Testing {num_hashes} hash functions (b={b}, r={r})...")
                
                map_score, _ = self.calculate_map_at_k_with_params(
                    mode=mode, shingle_k=baseline_params['shingle_k'], 
                    num_hashes=num_hashes, b=b, r=r, k=10
                )
                
                results[mode]['num_hashes'][num_hashes] = map_score
                print(f"✅ Hashes={num_hashes}: MAP@10 = {map_score:.4f}")
                
                # Save incremental results
                self._save_incremental_results(results, output_dir, f"{mode}_num_hashes_progress")
            
            # Generate intermediate table for hash results
            self._generate_intermediate_table(results[mode]['num_hashes'], 
                                            f"{output_dir}/{mode}_hash_functions_results.csv",
                                            "Num_Hashes", "Hash Functions Results")
            
            # 3. Test LSH parameters
            print(f"\n⚙️  TESTING LSH PARAMETERS ({mode})")
            print("-" * 50)
            
            for b, r in lsh_configs:
                current_test += 1
                print(f"[{current_test}/{total_tests}] Testing b={b}, r={r}...")
                
                map_score, _ = self.calculate_map_at_k_with_params(
                    mode=mode, shingle_k=baseline_params['shingle_k'], 
                    num_hashes=100, b=b, r=r, k=10
                )
                
                results[mode]['lsh_params'][(b, r)] = map_score
                print(f"✅ b={b}, r={r}: MAP@10 = {map_score:.4f}")
                
                # Save incremental results
                self._save_incremental_results(results, output_dir, f"{mode}_lsh_params_progress")
            
            # Generate intermediate table for LSH results
            lsh_data = {f"b={b},r={r}": score for (b, r), score in results[mode]['lsh_params'].items()}
            self._generate_intermediate_table(lsh_data, 
                                            f"{output_dir}/{mode}_lsh_params_results.csv",
                                            "LSH_Params", "LSH Parameters Results")
            
            # Generate intermediate summary for this mode
            self._generate_intermediate_summary(results[mode], mode, output_dir)
            
            print(f"\n🎯 COMPLETED MODE: {mode}")
            if results[mode]['shingle_k']:
                best_k = max(results[mode]['shingle_k'], key=results[mode]['shingle_k'].get)
                print(f"  - Best Shingle K: {best_k} (MAP@10: {results[mode]['shingle_k'][best_k]:.4f})")
            if results[mode]['num_hashes']:
                best_hash = max(results[mode]['num_hashes'], key=results[mode]['num_hashes'].get)
                print(f"  - Best Hash Count: {best_hash} (MAP@10: {results[mode]['num_hashes'][best_hash]:.4f})")
            if results[mode]['lsh_params']:
                best_lsh = max(results[mode]['lsh_params'], key=results[mode]['lsh_params'].get)
                print(f"  - Best LSH Params: {best_lsh} (MAP@10: {results[mode]['lsh_params'][best_lsh]:.4f})")
        
        # Generate final comprehensive results
        print(f"\n🏁 GENERATING FINAL COMPREHENSIVE RESULTS...")
        self.generate_tables_and_graphs(results)
        
        print(f"\n✅ EVALUATION COMPLETE!")
        print(f"📁 Incremental results saved in: {output_dir}/")
        print(f"📁 Final results saved in: exercise3_results/")
        
        return results
    
    def _save_incremental_results(self, results, output_dir, filename):
        """Save incremental results to JSON file"""
        import json
        
        # Convert tuple keys to strings for JSON serialization
        json_results = {}
        for mode in results:
            json_results[mode] = {}
            for test_type in results[mode]:
                json_results[mode][test_type] = {}
                for key, value in results[mode][test_type].items():
                    # Convert tuple keys to strings
                    if isinstance(key, tuple):
                        str_key = f"b={key[0]},r={key[1]}"
                    else:
                        str_key = str(key)
                    json_results[mode][test_type][str_key] = value
        
        with open(f"{output_dir}/{filename}.json", 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def _generate_intermediate_table(self, data, filepath, param_name, title):
        """Generate intermediate CSV table"""
        df = pd.DataFrame(list(data.items()), columns=[param_name, 'MAP@10'])
        df.to_csv(filepath, index=False)
        print(f"📊 Saved intermediate table: {filepath}")
    
    def _generate_intermediate_summary(self, mode_results, mode, output_dir):
        """Generate intermediate summary for a mode"""
        summary = f"\n{'='*40}\n"
        summary += f"INTERMEDIATE SUMMARY - {mode} MODE\n"
        summary += f"{'='*40}\n\n"
        
        if mode_results['shingle_k']:
            best_k = max(mode_results['shingle_k'], key=mode_results['shingle_k'].get)
            summary += f"Best Shingle K: {best_k} (MAP@10: {mode_results['shingle_k'][best_k]:.4f})\n"
            summary += f"Shingle K results: {dict(mode_results['shingle_k'])}\n\n"
        
        if mode_results['num_hashes']:
            best_hash = max(mode_results['num_hashes'], key=mode_results['num_hashes'].get)
            summary += f"Best Hash Count: {best_hash} (MAP@10: {mode_results['num_hashes'][best_hash]:.4f})\n"
            summary += f"Hash count results: {dict(mode_results['num_hashes'])}\n\n"
        
        if mode_results['lsh_params']:
            best_lsh = max(mode_results['lsh_params'], key=mode_results['lsh_params'].get)
            summary += f"Best LSH Params: {best_lsh} (MAP@10: {mode_results['lsh_params'][best_lsh]:.4f})\n"
            summary += f"LSH params results: {dict(mode_results['lsh_params'])}\n\n"
        
        with open(f"{output_dir}/{mode}_intermediate_summary.txt", 'w') as f:
            f.write(summary)
        
        print(summary)
    
    def generate_tables_and_graphs(self, results):
        """Generate tables and graphs for Exercise 3"""
        print("\nGenerating tables and graphs...")
        
        # Create output directory
        output_dir = "exercise3_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate tables
        self._generate_tables(results, output_dir)
        
        # Generate graphs
        self._generate_graphs(results, output_dir)
        
        # Generate summary report
        self._generate_summary_report(results, output_dir)
        
        print(f"All results saved in '{output_dir}' directory")
        return output_dir
    
    def _generate_tables(self, results, output_dir):
        """Generate CSV tables for all results"""
        
        for mode in ['PST', 'PSD']:
            # 1. Shingle K results
            shingle_data = []
            for k, map_score in results[mode]['shingle_k'].items():
                shingle_data.append({'K': k, 'MAP@10': map_score})
            
            df_shingle = pd.DataFrame(shingle_data)
            df_shingle.to_csv(f"{output_dir}/shingle_k_results_{mode}.csv", index=False)
            
            # 2. Hash functions results
            hash_data = []
            for num_hashes, map_score in results[mode]['num_hashes'].items():
                hash_data.append({'Num_Hashes': num_hashes, 'MAP@10': map_score})
            
            df_hash = pd.DataFrame(hash_data)
            df_hash.to_csv(f"{output_dir}/hash_functions_results_{mode}.csv", index=False)
            
            # 3. LSH parameters results
            lsh_data = []
            for (b, r), map_score in results[mode]['lsh_params'].items():
                lsh_data.append({'b': b, 'r': r, 'MAP@10': map_score})
            
            df_lsh = pd.DataFrame(lsh_data)
            df_lsh.to_csv(f"{output_dir}/lsh_params_results_{mode}.csv", index=False)
    
    def _generate_graphs(self, results, output_dir):
        """Generate visualization graphs"""
        
        # Set matplotlib backend to non-GUI to avoid threading issues
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Exercise 3: LSH Hyperparameter Evaluation Results', fontsize=16, fontweight='bold')
        
        modes = ['PST', 'PSD']
        mode_colors = ['blue', 'red']
        
        # 1. Shingle K comparison
        ax1 = axes[0, 0]
        for i, mode in enumerate(modes):
            k_values = list(results[mode]['shingle_k'].keys())
            k_scores = list(results[mode]['shingle_k'].values())
            ax1.plot(k_values, k_scores, 'o-', color=mode_colors[i], label=mode, linewidth=2, markersize=8)
        
        ax1.set_xlabel('K (Shingle Size)', fontweight='bold')
        ax1.set_ylabel('MAP@10', fontweight='bold')
        ax1.set_title('Effect of K-character Shingles', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Number of hashes comparison
        ax2 = axes[0, 1]
        for i, mode in enumerate(modes):
            hash_values = list(results[mode]['num_hashes'].keys())
            hash_scores = list(results[mode]['num_hashes'].values())
            ax2.plot(hash_values, hash_scores, 's-', color=mode_colors[i], label=mode, linewidth=2, markersize=8)
        
        ax2.set_xlabel('Number of Hash Functions', fontweight='bold')
        ax2.set_ylabel('MAP@10', fontweight='bold')
        ax2.set_title('Effect of Hash Functions', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. LSH parameters comparison
        ax3 = axes[0, 2]
        for i, mode in enumerate(modes):
            lsh_labels = [f"b={b},r={r}" for (b, r) in results[mode]['lsh_params'].keys()]
            lsh_scores = list(results[mode]['lsh_params'].values())
            x_pos = np.arange(len(lsh_labels)) + i * 0.35
            ax3.bar(x_pos, lsh_scores, width=0.35, color=mode_colors[i], alpha=0.7, label=mode)
        
        ax3.set_xlabel('LSH Parameters (b, r)', fontweight='bold')
        ax3.set_ylabel('MAP@10', fontweight='bold')
        ax3.set_title('Effect of LSH Parameters', fontweight='bold')
        ax3.set_xticks(np.arange(len(lsh_labels)) + 0.175)
        ax3.set_xticklabels(lsh_labels, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. PST detailed analysis
        ax4 = axes[1, 0]
        pst_all_scores = (list(results['PST']['shingle_k'].values()) + 
                         list(results['PST']['num_hashes'].values()) + 
                         list(results['PST']['lsh_params'].values()))
        ax4.hist(pst_all_scores, bins=15, alpha=0.7, color='blue', edgecolor='black')
        ax4.set_xlabel('MAP@10 Scores', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('PST: Distribution of MAP@10 Scores', fontweight='bold')
        ax4.axvline(np.mean(pst_all_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(pst_all_scores):.3f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. PSD detailed analysis
        ax5 = axes[1, 1]
        psd_all_scores = (list(results['PSD']['shingle_k'].values()) + 
                         list(results['PSD']['num_hashes'].values()) + 
                         list(results['PSD']['lsh_params'].values()))
        ax5.hist(psd_all_scores, bins=15, alpha=0.7, color='red', edgecolor='black')
        ax5.set_xlabel('MAP@10 Scores', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('PSD: Distribution of MAP@10 Scores', fontweight='bold')
        ax5.axvline(np.mean(psd_all_scores), color='blue', linestyle='--', 
                   label=f'Mean: {np.mean(psd_all_scores):.3f}')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Best configurations comparison
        ax6 = axes[1, 2]
        
        # Find best configuration for each category and mode
        best_configs = {}
        for mode in results:
            if any(results[mode].values()):  # Only process modes with results
                best_configs[mode] = {}
                if results[mode]['shingle_k']:
                    best_configs[mode]['shingle_k'] = max(results[mode]['shingle_k'], key=results[mode]['shingle_k'].get)
                if results[mode]['num_hashes']:
                    best_configs[mode]['num_hashes'] = max(results[mode]['num_hashes'], key=results[mode]['num_hashes'].get)
                if results[mode]['lsh_params']:
                    best_configs[mode]['lsh_params'] = max(results[mode]['lsh_params'], key=results[mode]['lsh_params'].get)
        
        if not best_configs:
            print("No results to plot!")
            return
        
        # Bar width
        width = 0.2
        
        # X locations for groups
        x_locations = np.arange(len(best_configs['PST']))
        
        # 1. Shingle K
        ax6.bar(x_locations, 
               [best_configs['PST']['shingle_k'], best_configs['PSD']['shingle_k']], 
               width=width, label='Shingle K', color='skyblue')
        
        # 2. Number of Hashes
        ax6.bar(x_locations + width, 
               [best_configs['PST']['num_hashes'], best_configs['PSD']['num_hashes']], 
               width=width, label='Num Hashes', color='lightgreen')
        
        # 3. LSH Parameters (b,r)
        lsh_labels = [f"b={b},r={r}" for (b, r) in results['PST']['lsh_params'].keys()]
        ax6.bar(x_locations + 2*width, 
               [results['PST']['lsh_params'][(best_configs['PST']['lsh_params'])], 
                results['PSD']['lsh_params'][(best_configs['PSD']['lsh_params'])]], 
               width=width, label='LSH Params', color='salmon')
        
        ax6.set_xticks(x_locations + width)
        ax6.set_xticklabels(['PST', 'PSD'])
        ax6.set_xlabel('Mode', fontweight='bold')
        ax6.set_ylabel('Best Configuration Values', fontweight='bold')
        ax6.set_title('Best Hyperparameter Configurations', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figures
        fig.savefig(f"{output_dir}/exercise3_results_summary.png")
        plt.close()  # Close figure instead of showing
    
    def _generate_summary_report(self, results, output_dir):
        """Generate a summary report (TXT) for the results"""
        print("\nGenerating summary report...")
        report_lines = []
        
        for mode in ['PST', 'PSD']:
            report_lines.append(f"=== Mode: {mode} ===")
            
            # Best Shingle K
            best_shingle_k = max(results[mode]['shingle_k'], key=results[mode]['shingle_k'].get)
            report_lines.append(f"Best Shingle K: {best_shingle_k} (MAP@10: {results[mode]['shingle_k'][best_shingle_k]:.4f})")
            
            # Best Number of Hashes
            best_num_hashes = max(results[mode]['num_hashes'], key=results[mode]['num_hashes'].get)
            report_lines.append(f"Best Number of Hashes: {best_num_hashes} (MAP@10: {results[mode]['num_hashes'][best_num_hashes]:.4f})")
            
            # Best LSH Params
            best_lsh_params = max(results[mode]['lsh_params'], key=results[mode]['lsh_params'].get)
            report_lines.append(f"Best LSH Params (b, r): {best_lsh_params} (MAP@10: {results[mode]['lsh_params'][best_lsh_params]:.4f})")
        
        # Save report to TXT file
        report_path = f"{output_dir}/exercise3_summary_report.txt"
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write("\n".join(report_lines))
        
        print(f"Summary report saved to: {report_path}")
