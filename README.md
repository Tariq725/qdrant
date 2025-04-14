# Hybrid Document Search Engine

A document search engine that combines both semantic (meaning-based) and keyword-based search capabilities, with support for fuzzy matching and typo tolerance.

## Features

- **Hybrid Search**: Balance between keyword and semantic search with an adjustable slider
- **Fuzzy Matching**: Find documents even with typos or slight variations in spelling
- **CSV Upload**: Upload CSV files with your documents for instant searching
- **Interactive UI**: User-friendly interface with real-time result highlighting
- **Diagnostics**: View search details for better understanding of results

## Getting Started

### Prerequisites

- Python 3.7+
- FastAPI
- scikit-learn
- numpy
- pandas

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hybrid-search-engine.git
cd hybrid-search-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:8000`

## Usage

1. Upload a CSV file containing your documents
2. Enter a search query
3. Adjust the search parameters:
   - **Search Balance**: Control the weight between keyword and semantic search
   - **Fuzzy Matching**: Enable to find results with slight spelling variations
   - **Fuzzy Threshold**: Control how strict the fuzzy matching should be

## How It Works

The engine combines two search approaches:
- **Keyword Search**: Finds documents containing the exact words in your query
- **Semantic Search**: Uses vector embeddings to find documents with similar meaning
- **Hybrid Approach**: Combines both methods with an adjustable weight

## License

MIT 