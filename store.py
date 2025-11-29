import chromadb
from sentence_transformers import SentenceTransformer
from ingest import extract_transcript # We import your previous work!
import os
import uuid

# 1. Setup the "Brain" (Embedding Model)
# This model is small, fast, and great for English text
print("Loading Embedding Model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Setup the Memory (Vector Database)
# This saves the DB to a folder named 'chroma_db' in your project
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="video_knowledge")

def add_to_vector_db(segments, video_name):
    print(f"Embedding {len(segments)} segments into the database...")
    
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for segment in segments:
        text = segment['text'].strip()
        start = segment['start']
        end = segment['end']
        
        # Skip empty segments
        if len(text) < 5:
            continue

        # Convert Text -> Vector
        vector = embedding_model.encode(text).tolist()
        
        # Prepare data for Chroma
        ids.append(str(uuid.uuid4())) # Unique ID for each chunk
        documents.append(text)
        embeddings.append(vector)
        # Metadata is CRITICAL: It tells us WHERE in the video to jump to
        metadatas.append({
            "video_name": video_name,
            "start_time": start,
            "end_time": end
        })

    # Save to DB in one big batch
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"✅ Success! Stored {len(documents)} chunks in ChromaDB.")

if __name__ == "__main__":
    video_path = "data/test_video.mp4"
    
    if os.path.exists(video_path):
        # Step 1: Get Text (using your ingest.py script)
        segments = extract_transcript(video_path)
        
        # Step 2: Save to DB
        add_to_vector_db(segments, "test_video")
    else:
        print("Video file not found.")