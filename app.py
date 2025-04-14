import re
import io
import uuid
import csv
import numpy as np
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="CSV Semantic Search")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for documents and vectors
DOCUMENTS: Dict[str, List[str]] = {}  # collection_id -> list of documents
VECTORS: Dict[str, Dict[str, Any]] = {}  # collection_id -> TF-IDF vectorizer and matrix

# Pydantic Models
class Query(BaseModel):
    collection_id: str
    query_text: str
    top_k: int = 5
    fuzzy_threshold: float = 0.7  # Threshold for fuzzy matching; values below 1.0 enable fuzzy matching
    hybrid_weight: float = 0.5  # 0 = pure keyword, 1 = pure semantic

class SearchResult(BaseModel):
    content: str
    score: float
    raw_score: float = 0.0
    metadata: Dict[str, Any] = {}

# Helper functions
def preprocess_text(text: str) -> str:
    """
    Preprocess text by converting to lowercase, removing punctuation, and normalizing spaces.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def expand_query_with_fuzzy_detailed(
    query_text: str, vocabulary: List[str], threshold: float
) -> Tuple[str, Dict[str, List[Tuple[str, float]]]]:
    """
    Expand the query using fuzzy matching over a given vocabulary.
    
    Returns:
        expanded_query (str): The expanded query string including fuzzy-matched words.
        fuzzy_matches (Dict[str, List[Tuple[str, float]]]): Maps each original word to a list
            of tuples (vocab_word, similarity) that exceeded the threshold.
    """
    original_words = preprocess_text(query_text).split()
    expanded_words = set(original_words)
    fuzzy_matches: Dict[str, List[Tuple[str, float]]] = {}
    
    for word in original_words:
        if len(word) <= 2:
            continue
        fuzzy_matches[word] = []
        for vocab_word in vocabulary:
            if len(vocab_word) <= 2 or vocab_word == word:
                continue
            similarity = SequenceMatcher(None, word, vocab_word).ratio()
            adjusted_threshold = threshold
            # Increase threshold for shorter words to reduce false positives
            if len(word) <= 4:
                adjusted_threshold = max(threshold + 0.1, 0.8)
            if similarity >= adjusted_threshold:
                expanded_words.add(vocab_word)
                fuzzy_matches[word].append((vocab_word, similarity))
    return ' '.join(expanded_words), fuzzy_matches

def get_html() -> str:
    """
    Return the HTML content for the home page.
    """
    return """<!DOCTYPE html>
<html>
<head>
    <title>CSV Semantic Search</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .section {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
        }
        .result {
            border: 1px solid #eee;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        button {
            padding: 8px 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        input, textarea {
            width: 100%;
            padding: 8px;
            margin: 8px 0;
            box-sizing: border-box;
        }
        .info {
            background-color: #e7f3fe;
            border-left: 6px solid #2196F3;
            padding: 10px;
            margin: 10px 0;
        }
        .success {
            background-color: #e7f3e8;
            border-left: 6px solid #4CAF50;
            padding: 10px;
            margin: 10px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .options {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .checkbox-container {
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }
        .checkbox-container input[type="checkbox"] {
            width: auto;
            margin-right: 10px;
        }
        .slider-container {
            display: flex;
            flex-direction: column;
            margin-top: 10px;
        }
        .range-value {
            margin-top: 5px;
            text-align: center;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            margin-left: 5px;
            cursor: help;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 300px;
            background-color: #555;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.9em;
            line-height: 1.4;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        .match-details {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            padding: 8px;
            margin-top: 10px;
            border-radius: 4px;
        }
        .boost-info {
            color: #2196F3;
            font-style: italic;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>CSV Semantic Search</h1>
    
    <div class="container">
        <div class="section">
            <h2>Step 1: Upload CSV</h2>
            <p class="info">Upload a CSV file. The first column will be used for semantic search.</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="csvFile" accept=".csv" required>
                <button type="submit">Upload and Process</button>
            </form>
            <div id="uploadResult"></div>
        </div>

        <div class="section">
            <h2>Step 2: Semantic Search</h2>
            <form id="searchForm">
                <input type="text" id="collectionId" placeholder="Collection ID" required>
                <textarea id="queryInput" placeholder="Enter your search query" rows="3" required></textarea>
                <input type="number" id="topK" placeholder="Number of results" value="5" min="1" max="20">
                
                <div class="options">
                    <h3>Advanced Options</h3>
                    <div class="checkbox-container">
                        <input type="checkbox" id="enableFuzzy" checked>
                        <label for="enableFuzzy">Enable typo tolerance (fuzzy matching)</label>
                        <div class="tooltip">ⓘ
                            <span class="tooltiptext">
                                When enabled, the search will find results even when you have typos in your query. 
                                The system will intelligently boost scores for documents that match similar words.
                                <br><br>
                                Example: Searching for "reccomendation" will also match "recommendation" with a boosted score.
                            </span>
                        </div>
                    </div>
                    
                    <div class="slider-container">
                        <label for="fuzzyThreshold">Fuzzy matching threshold:</label>
                        <input type="range" id="fuzzyThreshold" min="0.5" max="0.9" value="0.7" step="0.05">
                        <div class="range-value" id="thresholdValue">0.7</div>
                        <div class="tooltip">ⓘ
                            <span class="tooltiptext">
                                Controls how similar words need to be to be considered a match:<br>
                                - Higher values (0.8-0.9): Stricter matching, only very similar words<br>
                                - Medium values (0.7): Balanced approach<br>
                                - Lower values (0.5-0.6): More lenient matching, captures more variations but might include unrelated terms
                            </span>
                        </div>
                    </div>
                    
                    <div class="slider-container">
                        <label for="hybridWeight">Search balance:</label>
                        <input type="range" id="hybridWeight" min="0" max="1" value="0.5" step="0.1">
                        <div class="range-value" id="hybridWeightValue">0.5</div>
                        <div class="tooltip">ⓘ
                            <span class="tooltiptext">
                                Controls the balance between keyword search and semantic search:<br>
                                - 0: Pure keyword search (exact matches)<br>
                                - 0.5: Balanced hybrid search<br>
                                - 1: Pure semantic search (meaning)
                            </span>
                        </div>
                    </div>
                </div>
                
                <button type="submit">Search</button>
            </form>
        </div>

        <div class="section">
            <h2>Results</h2>
            <div id="searchResults"></div>
            <div id="matchDetails" class="match-details" style="display: none;"></div>
        </div>
    </div>

    <script>
        // Show the current threshold value
        const thresholdSlider = document.getElementById('fuzzyThreshold');
        const thresholdValue = document.getElementById('thresholdValue');
        thresholdSlider.addEventListener('input', () => {
            thresholdValue.textContent = thresholdSlider.value;
        });
        
        // Show the current hybrid weight value
        const hybridSlider = document.getElementById('hybridWeight');
        const hybridValue = document.getElementById('hybridWeightValue');
        hybridSlider.addEventListener('input', () => {
            hybridValue.textContent = hybridSlider.value;
        });
        
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('csvFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a CSV file');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                document.getElementById('uploadResult').innerHTML = '<p>Uploading and processing CSV...</p>';
                
                const response = await fetch('/upload-csv/', {
                    method: 'POST',
                    body: formData,
                });
                
                const result = await response.json();
                if (result.error) {
                    document.getElementById('uploadResult').innerHTML = `<p>Error: ${result.error}</p>`;
                    return;
                }
                
                let tableHtml = `
                    <div class="success">
                        <p>CSV file uploaded and processed successfully!</p>
                        <p><strong>Collection ID:</strong> ${result.collection_id}</p>
                        <p><strong>Documents processed:</strong> ${result.document_count}</p>
                        <p>Use the Collection ID above for searching.</p>
                    </div>
                `;
                
                if (result.sample_documents && result.sample_documents.length > 0) {
                    tableHtml += '<p><strong>Sample of processed documents:</strong></p>';
                    tableHtml += '<table><tr><th>Content</th></tr>';
                    result.sample_documents.forEach(doc => {
                        tableHtml += `<tr><td>${doc}</td></tr>`;
                    });
                    tableHtml += '</table>';
                }
                
                document.getElementById('uploadResult').innerHTML = tableHtml;
                document.getElementById('collectionId').value = result.collection_id;
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('uploadResult').innerHTML = `<p>Error uploading file: ${error.message}</p>`;
            }
        });

        document.getElementById('searchForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const collectionId = document.getElementById('collectionId').value;
            const query = document.getElementById('queryInput').value;
            const topK = document.getElementById('topK').value;
            const enableFuzzy = document.getElementById('enableFuzzy').checked;
            const fuzzyThreshold = document.getElementById('fuzzyThreshold').value;
            const hybridWeight = document.getElementById('hybridWeight').value;
            
            if (!collectionId || !query) {
                alert('Please enter both collection ID and query');
                return;
            }
            
            try {
                document.getElementById('searchResults').innerHTML = '<p>Searching...</p>';
                document.getElementById('matchDetails').style.display = 'none';
                
                const response = await fetch('/search/', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        collection_id: collectionId,
                        query_text: query,
                        top_k: parseInt(topK),
                        fuzzy_threshold: enableFuzzy ? parseFloat(fuzzyThreshold) : 1.0,
                        hybrid_weight: parseFloat(hybridWeight)
                    }),
                });
                
                const result = await response.json();
                if (result.detail) {
                    document.getElementById('searchResults').innerHTML = `<p>Error: ${result.detail}</p>`;
                    return;
                }
                
                if (result.results.length === 0) {
                    document.getElementById('searchResults').innerHTML = '<p>No results found for your query.</p>';
                    return;
                }
                
                const resultsHtml = result.results.map((item, index) => `
                    <div class="result">
                        <h3>Result ${index + 1}</h3>
                        <p>${item.content}</p>
                        <p style="color: #888;">Normalized Score: ${item.score.toFixed(4)} (Raw Score: ${item.raw_score.toFixed(4)})</p>
                    </div>
                `).join('');
                
                document.getElementById('searchResults').innerHTML = resultsHtml;
                if (enableFuzzy && result.diagnostics && 
                    result.diagnostics.original_query !== result.diagnostics.expanded_query) {
                    document.getElementById('matchDetails').style.display = 'block';
                    document.getElementById('matchDetails').innerHTML = `
                        <h3>Query Expansion Details</h3>
                        <p><strong>Original Query:</strong> ${result.diagnostics.original_query}</p>
                        <p><strong>Expanded Query:</strong> ${result.diagnostics.expanded_query}</p>
                        <p class="boost-info">* Scores have been boosted for documents that match similar words</p>
                    `;
                }
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('searchResults').innerHTML = `<p>Error searching: ${error.message}</p>`;
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_index() -> str:
    """
    Endpoint to serve the home page.
    """
    return get_html()

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload and process a CSV file.
    The first column of each row is treated as the document content.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        contents = await file.read()
        csv_file = io.StringIO(contents.decode('utf-8-sig'))
        csv_reader = csv.reader(csv_file)
        # Skip header row
        next(csv_reader, None)
        documents = []
        for row in csv_reader:
            if row and row[0].strip():
                content = row[0].strip()
                # Remove surrounding quotes if present
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                if content:
                    documents.append(content)
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents found in the CSV file")
        collection_id = str(uuid.uuid4())
        DOCUMENTS[collection_id] = documents

        vectorizer = TfidfVectorizer(
            stop_words='english',
            sublinear_tf=True,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2)
        )
        matrix = vectorizer.fit_transform(documents)
        VECTORS[collection_id] = {'vectorizer': vectorizer, 'matrix': matrix}
        
        return {
            'collection_id': collection_id,
            'document_count': len(documents),
            'sample_documents': documents[:5]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {str(e)}")

@app.post("/search/")
async def search(query: Query):
    """
    Perform hybrid (keyword + semantic) search on documents with optional fuzzy matching.
    """
    collection_id = query.collection_id
    original_query_text = query.query_text
    top_k = min(query.top_k, 20)
    fuzzy_threshold = query.fuzzy_threshold
    hybrid_weight = max(0.0, min(1.0, query.hybrid_weight))
    
    if collection_id not in DOCUMENTS or collection_id not in VECTORS:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    try:
        documents = DOCUMENTS[collection_id]
        vectorizer = VECTORS[collection_id]['vectorizer']
        document_matrix = VECTORS[collection_id]['matrix']
        
        fuzzy_matches: Dict[str, List[Tuple[str, float]]] = {}
        # Expand query with fuzzy matching if enabled
        if fuzzy_threshold < 1.0:
            vocabulary = vectorizer.get_feature_names_out().tolist()
            expanded_query, fuzzy_matches = expand_query_with_fuzzy_detailed(original_query_text, vocabulary, fuzzy_threshold)
            query_text = expanded_query
        else:
            query_text = original_query_text
        
        query_vector = vectorizer.transform([query_text])
        semantic_similarities = cosine_similarity(query_vector, document_matrix).flatten()
        
        # Calculate keyword scores from the original query
        keyword_scores = np.zeros(len(documents))
        query_words = set(preprocess_text(original_query_text).split())
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            doc_words = set(preprocess_text(doc_lower).split())
            if query_words:
                matches = query_words.intersection(doc_words)
                keyword_scores[i] = len(matches) / len(query_words)
            if original_query_text.lower() in doc_lower:
                keyword_scores[i] += 0.5
        
        if np.max(keyword_scores) > 0:
            keyword_scores = keyword_scores / np.max(keyword_scores)
        
        # Combine semantic and keyword scores with optional boost factors
        similarities = np.zeros(len(documents))
        for i, doc in enumerate(documents):
            base_score = (1 - hybrid_weight) * keyword_scores[i] + hybrid_weight * semantic_similarities[i]
            boost_score = 0.0
            doc_lower = doc.lower()
            
            # Boost for exact query match
            if original_query_text.lower() in doc_lower:
                boost_score += 0.4
            # Boost for each query word as a whole word match
            for word in preprocess_text(original_query_text).split():
                if len(word) > 2:
                    word_length_factor = min(len(word) / 5, 1.0)
                    if re.search(r'\b' + re.escape(word) + r'\b', doc_lower):
                        boost_score += 0.2 * word_length_factor
            # Boost for fuzzy matches if enabled
            if fuzzy_threshold < 1.0:
                for word, matches in fuzzy_matches.items():
                    if word in doc_lower:
                        continue
                    for match_word, similarity in matches:
                        if match_word in doc_lower:
                            boost_score += 0.1 * similarity
                            break
            final_score = min(0.6 * base_score + 0.4 * min(boost_score, 1.0), 1.0)
            similarities[i] = final_score
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = [
            SearchResult(
                content=documents[idx],
                score=float(similarities[idx]),
                raw_score=float(similarities[idx]),
                metadata={}
            )
            for idx in top_indices if similarities[idx] > 0
        ]
        diagnostics = {
            'original_query': original_query_text,
            'expanded_query': query_text if fuzzy_threshold < 1.0 else original_query_text,
            'fuzzy_threshold': fuzzy_threshold,
            'hybrid_weight': hybrid_weight,
            'max_score': float(np.max(similarities)) if similarities.size > 0 else 0.0
        }
        
        return {'results': results, 'diagnostics': diagnostics}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
