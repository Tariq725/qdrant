# CSV Semantic Search

A web application that allows users to upload CSV files and perform semantic search on the content of the first column.

## Features

- Upload CSV files and process them for semantic search
- Hybrid search combining keyword and semantic matching
- Fuzzy matching to handle typos and similar words
- Adjustable search parameters:
  - Fuzzy matching threshold
  - Hybrid search weight (keyword vs. semantic)
  - Number of results to return

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:

```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:8000`
3. Upload a CSV file (the first column will be used for semantic search)
4. Use the provided Collection ID to perform searches

## API Endpoints

- `GET /`: Serves the web interface
- `POST /upload-csv/`: Upload and process a CSV file
- `POST /search/`: Perform semantic search on processed documents

## Search Parameters

- `collection_id`: ID of the document collection to search
- `query_text`: Search query text
- `top_k`: Number of results to return (default: 5, max: 20)
- `fuzzy_threshold`: Threshold for fuzzy matching (default: 0.7)
- `hybrid_weight`: Weight for hybrid search (0 = pure keyword, 1 = pure semantic, default: 0.5)

## Technologies Used

- FastAPI: Web framework
- scikit-learn: TF-IDF vectorization and cosine similarity
- NumPy: Numerical operations
- HTML/CSS/JavaScript: Frontend interface

## License

MIT 