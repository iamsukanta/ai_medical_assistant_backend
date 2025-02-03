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

# Initialize FAISS Index
vector_dim = 384  # Model Output Dimension
index = faiss.IndexFlatL2(vector_dim)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["medical_db"]
collection = db["patient_queries"]

# For Vector DB
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# For API Call
API_KEY = os.getenv("HUGGINGFACE_API_KEY");
headers = {"Authorization": f"Bearer {API_KEY}"}

@app.route("/search", methods=["POST"])
def search_medical_assistant():
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Generate AI response
    response = response = requests.post(
        "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
        headers=headers,
        json={"inputs": user_query },
    )

    # Store query and response in MongoDB
    collection.insert_one({"query": user_query, "response": response.json()})
    return jsonify({"response": response.json()})

if __name__ == "__main__":
    app.run(debug=True)
