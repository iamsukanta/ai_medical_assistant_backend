import os
import requests
import faiss
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


app = Flask(__name__)
CORS(app)

load_dotenv()

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["medical_db"]
collection = db["patient_queries"]

# For Vector DB
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# FAISS Vector Storage (Assume 384-dimension vectors)
vector_dim = 384
index = faiss.IndexFlatL2(vector_dim)

# Store Metadata (Query Texts)
query_metadata = []

# For API Call
API_KEY = os.getenv("HUGGINGFACE_API_KEY");
headers = {"Authorization": f"Bearer {API_KEY}"}

def normalize(vectors):
    """Normalize vectors for cosine similarity."""
    faiss.normalize_L2(vectors)
    return vectors

@app.route("/search", methods=["POST"])
def search_medical_assistant():
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Convert Query to Vector
    vector = embed_model.encode([user_query], normalize_embeddings=True)
    vector = np.array(vector, dtype="float32")

    # Search in FAISS
    D, I = index.search(vector, k=3)

    # Get matched documents from MongoDB
    matched_results = []
    for idx in I[0]:  # Get top results
        if idx == -1:
            continue  # Ignore invalid results
        result = collection.find_one({"query": {"$in": [user_query]}},  {"_id": 0})
        if result:
            matched_results.append(result)
            break;

    if len(matched_results) and len(I[0]):
        return jsonify({"matches": matched_results[0]})
    else:
        # Generate AI response
        response = response = requests.post(
            "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
            headers=headers,
            json={"inputs": user_query },
        )

        # Store query and response in MongoDB
        collection.insert_one({"query": user_query, "response": response.json(), "vector": vector[0].tolist()})
        index.add(vector)
        return jsonify({"response": response.json()})

if __name__ == "__main__":
    app.run(debug=True)
