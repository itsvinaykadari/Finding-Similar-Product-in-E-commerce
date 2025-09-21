# Finding Similar Products in E-commerce

A Flask web application for finding similar products using LSH (Locality Sensitive Hashing) and MinHash algorithms on Amazon product data.

## Prerequisites

Before running the code, ensure you have the following:
- Python 3.8 or higher
- The `meta_Appliances.json` dataset file in the same folder as the code

## Setup and Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Code

Follow these steps in order:

### Step 1: Data Preprocessing
First, run the preprocessing script to clean and prepare the data:
```bash
python preprocess.py
```
**Important:** The `meta_Appliances.json` dataset file must be present in the same folder for this step to work.

This script will:
- Clean the product data
- Extract valid product ASINs from HTML content
- Generate `processed_appliances.json` file

### Step 2: Build LSH Index
Next, build the LSH index for similarity search:
```bash
python index_lsh.py
```

This script will:
- Create MinHash signatures for product titles and descriptions
- Build LSH indices for fast similarity search
- Generate `lsh_title.pkl` and `lsh_desc.pkl` files

### Step 3: Run the Web Application
Finally, start the Flask web application:
```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`

## Features

### Main Application
- **Product Search**: Find similar products based on title or description
- **ROUGE Score Evaluation**: Calculate similarity metrics between products
- **Interactive UI**: Clean and user-friendly web interface

### Exercise 3: LSH Hyperparameter Evaluation
The application includes a comprehensive evaluation system for LSH hyperparameters:

1. **Access Exercise 3**: In the web UI, click on the **"Exercise 3: LSH Hyperparameter Evaluation"** section
2. **Run Evaluation**: Click the evaluation button to start the hyperparameter testing
3. **View Results**: The evaluation results will be automatically saved in the `exercise3_results` folder

The Exercise 3 evaluation includes:
- Grid search over multiple hyperparameter combinations
- MAP@10 (Mean Average Precision at 10) calculations
- Performance comparison across different configurations
- Detailed results with precision, recall, and F1 scores

## File Descriptions

- `preprocess.py`: Data cleaning and processing functions for Amazon product data
- `index_lsh.py`: MinHash signature creation and LSH index building
- `query.py`: Query processing for finding similar products
- `app.py`: Flask web application with UI and API endpoints
- `exercise3_evaluation.py`: Comprehensive LSH hyperparameter evaluation system
- `requirements.txt`: List of required Python libraries
- `meta_Appliances.json`: Amazon product dataset (required input file)
- `processed_appliances.json`: Cleaned and processed product data (generated)
- `lsh_title.pkl` / `lsh_desc.pkl`: LSH index files (generated)

## Output Folders

- `exercise3_results/`: Contains evaluation results from Exercise 3 hyperparameter testing

## Dataset Requirements

The application requires the `meta_Appliances.json` file containing Amazon product data. This file should include:
- Product ASINs
- Product titles
- Product descriptions
- Similar product information

## Troubleshooting

1. **Missing dataset error**: Ensure `meta_Appliances.json` is in the same folder as the Python files
2. **Import errors**: Install all dependencies using `pip install -r requirements.txt`
3. **LSH index errors**: Make sure to run `preprocess.py` and `index_lsh.py` before starting the web app
4. **Exercise 3 results**: Check the `exercise3_results` folder for saved evaluation outputs
