import chromadb
from sentence_transformers import SentenceTransformer

# 1. Connect to the "Brain" (Load the existing database)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="video_knowledge")

# 2. Load the Translator (Text -> Vector Model)
# We must use the SAME model we used to store the data
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_video(query):
    print(f"\n🔎 Searching for: '{query}'...")
    
    # Convert question to vector
    query_vector = model.encode(query).tolist()
    
    # Ask the DB for the top 3 best matches
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )
    
    # Process results
    if not results['documents'][0]:
        print("No matches found.")
        return

    print("\n--- TOP MATCHES ---")
    for i in range(len(results['documents'][0])):
        text_match = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        start_time = metadata['start_time']
        confidence = results['distances'][0][i] # Lower is better in Chroma
        
        print(f"#{i+1} [Time: {start_time:.2f}s]: \"{text_match}\"")

if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a question about the video (or type 'exit'): ")
        if user_query.lower() == 'exit':
            break
        
        search_video(user_query)