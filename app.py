import streamlit as st
from streamlit_lottie import st_lottie
import requests
import chromadb
from sentence_transformers import SentenceTransformer
import os
import yt_dlp
import re
from groq import Groq
from dotenv import load_dotenv
from ingest import extract_transcript

# --- LOAD SECRETS ---
load_dotenv()
# Try to get API Key from .env or Streamlit Secrets (for when we deploy)
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY and "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = None

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="VideoRAG Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError: pass

local_css("assets/style.css")

# --- SESSION STATE ---
if "processed_video" not in st.session_state:
    st.session_state.processed_video = None

# --- ASSETS ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_processing = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_bty3111j.json")

# --- CACHED MODELS ---
@st.cache_resource
def setup_models():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="chroma_db")

embedding_model = setup_models()
chroma_client = get_chroma_client()
collection = chroma_client.get_or_create_collection(name="video_knowledge")

# --- LLM ANSWER GENERATION ---
def generate_ai_answer(context_text, user_question):
    if not GROQ_API_KEY:
        return "⚠️ Groq API Key not found. Cannot generate AI answer."
    
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
    Context from video transcript: "{context_text}"
    
    User Question: "{user_question}"
    
    Based ONLY on the context provided, answer the question clearly and concisely.
    """
    
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
    )
    return completion.choices[0].message.content

# --- YOUTUBE DOWNLOADER ---
def process_youtube_url(url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': 'data/%(id)s.%(ext)s',
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return os.path.join("data", f"{info['id']}.mp4"), info['title']

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📥 Add Content")
    
    # API KEY INPUT (If not in .env)
    if not GROQ_API_KEY:
        st.warning("⚠️ Groq API Key missing!")
        user_key = st.text_input("Enter Groq API Key", type="password")
        if user_key: GROQ_API_KEY = user_key

    tab1, tab2 = st.tabs(["📁 Upload", "🔗 YouTube"])
    video_path = None
    video_identifier = None

    with tab1:
        uploaded_file = st.file_uploader("Choose MP4", type=["mp4"])
        if uploaded_file:
            if not os.path.exists("data"): os.makedirs("data")
            save_path = os.path.join("data", uploaded_file.name)
            with open(save_path, "wb") as f: f.write(uploaded_file.getbuffer())
            video_path = save_path
            video_identifier = uploaded_file.name

    with tab2:
        youtube_url = st.text_input("Paste YouTube Link")
        if youtube_url and st.button("Download"):
            with st.spinner("Downloading..."):
                try:
                    if not os.path.exists("data"): os.makedirs("data")
                    path, title = process_youtube_url(youtube_url)
                    video_path = path
                    video_identifier = title
                    st.success(f"Downloaded: {title}")
                except Exception as e: st.error(f"Failed: {e}")

    # --- PROCESSING PIPELINE ---
    if video_path and video_identifier != st.session_state.processed_video:
        st.info(f"Processing: {video_identifier}")
        if lottie_processing: st_lottie(lottie_processing, height=150, key="proc")
        
        progress = st.progress(0)
        status = st.empty()
        
        try:
            status.text("👂 Transcribing...")
            progress.progress(20)
            segments = extract_transcript(video_path)
            
            status.text("🧠 Embedding...")
            progress.progress(60)
            
            ids, docs, metas, vecs = [], [], [], []
            for i, seg in enumerate(segments):
                text = seg['text'].strip()
                if len(text) < 5: continue
                safe_id = re.sub(r'\W+', '', video_identifier) + f"_{i}"
                vecs.append(embedding_model.encode(text).tolist())
                ids.append(safe_id)
                docs.append(text)
                metas.append({
                    "video_name": video_identifier,
                    "video_path": video_path,
                    "start_time": seg['start'],
                    "end_time": seg['end']
                })
            
            collection.add(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
            progress.progress(100)
            st.session_state.processed_video = video_identifier
            st.session_state.current_video_path = video_path
            st.rerun()
            
        except Exception as e: st.error(f"Error: {e}")

# --- MAIN UI ---
col1, col2 = st.columns([1, 5])
with col1:
    if lottie_ai: st_lottie(lottie_ai, height=80, key="logo")
with col2:
    st.markdown("<h1 style='padding-top: 10px;'>VideoRAG <span style='color:#FF4B4B'>Pro</span></h1>", unsafe_allow_html=True)

st.markdown("---")

if st.session_state.processed_video:
    vid_name = st.session_state.processed_video
    vid_path = st.session_state.get("current_video_path", "")
    
    st.markdown(f"#### 🎬 Watching: *{vid_name}*")
    
    query = st.text_input("", placeholder="Ask a question about the video...")
    
    if query:
        query_vec = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=1,
            where={"video_name": vid_name}
        )
        
        if not results['documents'][0]:
            st.warning("No answer found.")
        else:
            best_text = results['documents'][0][0]
            metadata = results['metadatas'][0][0]
            start_time = int(metadata['start_time'])
            
            # --- AI ANSWER SECTION ---
            with st.spinner("🤖 AI is thinking..."):
                ai_answer = generate_ai_answer(best_text, query)
            
            st.markdown(f"""
            <div style="background-color: #262730; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B; margin-bottom: 20px;">
                <h5 style="margin: 0; color: #FF4B4B;">🤖 AI Answer:</h5>
                <p style="margin-top: 5px; font-size: 16px;">{ai_answer}</p>
            </div>
            """, unsafe_allow_html=True)

            # --- VIDEO PLAYER ---
            st.success(f"Playing relevant clip at **{start_time}s**")
            if os.path.exists(vid_path):
                st.video(vid_path, start_time=start_time)
            
            # --- TRANSCRIPT ---
            st.markdown(f"<div class='result-card'><p style='color:#888; font-size:12px;'>SOURCE TRANSCRIPT</p><p>\"{best_text}\"</p></div>", unsafe_allow_html=True)
else:
    st.info("👈 Upload a video to start.")