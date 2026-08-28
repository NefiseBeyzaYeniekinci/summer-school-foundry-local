import streamlit as st
import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

st.set_page_config(page_title="Yerel RAG Mutfak Asistanı", page_icon="🍳")
st.title("🍳 Yerel RAG Mutfak Asistanı")
st.markdown("Bu asistan **Microsoft Foundry Local** ve yerel vektör arama (RAG) kullanarak çalışır. Tamamen internet bağlantısı olmadan, cihazınızda çalışır!")

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_top_recipes(query_embedding, db_path="recipes.db", top_k=3):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, chunk_text, embedding FROM recipes")
    rows = cursor.fetchall()
    
    scored_results = []
    for row in rows:
        db_id, title, chunk_text, emb_json = row
        db_embedding = json.loads(emb_json)
        score = cosine_similarity(query_embedding, db_embedding)
        scored_results.append((score, title, chunk_text))
    
    conn.close()
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return scored_results[:top_k]

def ask_groq(context_text, user_question, model_name="llama-3.1-8b-instant"):
    prompt = f"""
Sen yetenekli bir mutfak asistanısın. Aşağıdaki tarif bilgilerini kullanarak kullanıcının sorusuna cevap ver.
Eğer cevap bu metinlerde yoksa, "Bilmiyorum, sistemimde bu tarif yok" şeklinde dürüstçe yanıtla.

[BAĞLAM (TARİFLER)]:
{context_text}

[KULLANICI SORUSU]:
{user_question}
    """
    
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=st.secrets["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Sen yerel RAG mutfak asistanısın."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Yapay zeka ile iletişim kurulamadı.\n\nHata Detayı: {e}"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Kullanılan Kaynaklar"):
                st.write(msg["sources"])

if prompt := st.chat_input("Bana bir yemek tarifi sor... (Örn: How to make a chicken stew?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Tarifler aranıyor ve cevap üretiliyor..."):
            query_embedding = embedder.encode(prompt).tolist()
            top_results = get_top_recipes(query_embedding)
            
            context_text = ""
            sources_text = ""
            for i, (score, title, text) in enumerate(top_results):
                context_text += f"\n---\n{text}\n"
                sources_text += f"**{i+1}. {title}** (Benzerlik: %{int(score*100)})\n"
            
            answer = ask_groq(context_text, prompt)
            
            st.markdown(answer)
            with st.expander("Kullanılan Kaynaklar (Tıkla ve Aç)"):
                st.markdown(sources_text)
                
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_text})
