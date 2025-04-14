import re
import io
import uuid
import csv
import numpy as np
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI(title="CSV Semantic Search")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory storage for documents and vectors
DOCUMENTS: Dict[str, List[str]] = {}  # collection_id -> list of documents
VECTORS: Dict[str, Dict[str, Any]] = {}  # collection_id -> TF-IDF vectorizer and matrix

# Pydantic Models
class Query(BaseModel):
    """Query model for search requests."""
    collection_id: str = Field(..., description="ID of the document collection to search")
    query_text: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    fuzzy_threshold: float = Field(
        default=0.7, 
        ge=0.5, 
        le=1.0, 
        description="Threshold for fuzzy matching; values below 1.0 enable fuzzy matching"
    )
    hybrid_weight: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0, 
        description="Weight for hybrid search: 0 = pure keyword, 1 = pure semantic"
    )

class SearchResult(BaseModel):
    """Model for search results."""
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Normalized score")
    raw_score: float = Field(default=0.0, description="Raw score before normalization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

# Helper functions
def preprocess_text(text: str) -> str:
    """
    Preprocess text by converting to lowercase, removing punctuation, and normalizing spaces.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
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
    
    Args:
        query_text: Original query text
        vocabulary: List of vocabulary words to match against
        threshold: Similarity threshold for fuzzy matching
        
    Returns:
        Tuple containing:
        - expanded_query: The expanded query string including fuzzy-matched words
        - fuzzy_matches: Maps each original word to a list of tuples (vocab_word, similarity)
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

def calculate_keyword_scores(documents: List[str], query_text: str) -> np.ndarray:
    """
    Calculate keyword-based scores for documents based on query text.
    
    Args:
        documents: List of document texts
        query_text: Query text to match against
        
    Returns:
        Array of keyword scores for each document
    """
    keyword_scores = np.zeros(len(documents))
    query_words = set(preprocess_text(query_text).split())
    
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        doc_words = set(preprocess_text(doc_lower).split())
        if query_words:
            matches = query_words.intersection(doc_words)
            keyword_scores[i] = len(matches) / len(query_words)
        if query_text.lower() in doc_lower:
            keyword_scores[i] += 0.5
    
    if np.max(keyword_scores) > 0:
        keyword_scores = keyword_scores / np.max(keyword_scores)
    
    return keyword_scores

def calculate_boost_scores(
    doc: str, 
    original_query: str, 
    fuzzy_matches: Optional[Dict[str, List[Tuple[str, float]]]] = None
) -> float:
    """
    Calculate boost scores for a document based on various matching criteria.
    
    Args:
        doc: Document text
        original_query: Original query text
        fuzzy_matches: Optional dictionary of fuzzy matches
        
    Returns:
        Boost score for the document
    """
    boost_score = 0.0
    doc_lower = doc.lower()
    
    # Boost for exact query match
    if original_query.lower() in doc_lower:
        boost_score += 0.4
    
    # Boost for each query word as a whole word match
    for word in preprocess_text(original_query).split():
        if len(word) > 2:
            word_length_factor = min(len(word) / 5, 1.0)
            if re.search(r'\b' + re.escape(word) + r'\b', doc_lower):
                boost_score += 0.2 * word_length_factor
    
    # Boost for fuzzy matches if enabled
    if fuzzy_matches:
        for word, matches in fuzzy_matches.items():
            if word in doc_lower:
                continue
            for match_word, similarity in matches:
                if match_word in doc_lower:
                    boost_score += 0.1 * similarity
                    break
    
    return min(boost_score, 1.0)

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def get_index() -> str:
    """Serve the home page."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload and process a CSV file.
    
    The first column of each row is treated as the document content.
    
    Args:
        file: CSV file to upload
        
    Returns:
        Dictionary with collection_id, document_count, and sample_documents
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
    
    Args:
        query: Query object containing search parameters
        
    Returns:
        Dictionary with search results and diagnostics
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
        
        # Expand query with fuzzy matching if enabled
        fuzzy_matches = None
        if fuzzy_threshold < 1.0:
            vocabulary = vectorizer.get_feature_names_out().tolist()
            expanded_query, fuzzy_matches = expand_query_with_fuzzy_detailed(
                original_query_text, vocabulary, fuzzy_threshold
            )
            query_text = expanded_query
        else:
            query_text = original_query_text
        
        # Calculate semantic similarity
        query_vector = vectorizer.transform([query_text])
        semantic_similarities = cosine_similarity(query_vector, document_matrix).flatten()
        
        # Calculate keyword scores
        keyword_scores = calculate_keyword_scores(documents, original_query_text)
        
        # Combine semantic and keyword scores with boost factors
        similarities = np.zeros(len(documents))
        for i, doc in enumerate(documents):
            base_score = (1 - hybrid_weight) * keyword_scores[i] + hybrid_weight * semantic_similarities[i]
            boost_score = calculate_boost_scores(doc, original_query_text, fuzzy_matches)
            final_score = min(0.6 * base_score + 0.4 * boost_score, 1.0)
            similarities[i] = final_score
        
        # Get top results
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
        
        # Prepare diagnostics
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
