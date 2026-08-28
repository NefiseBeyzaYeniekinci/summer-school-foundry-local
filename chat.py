import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1. Cosine Similarity (Vektör Benzerliği) hesaplama fonksiyonu
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# 2. Veritabanından en alakalı tarifleri bulma
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
    
    # Skorlara göre büyükten küçüğe sıralayalım
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return scored_results[:top_k]

# 3. Microsoft Foundry Local Entegrasyonu (OpenAI uyumlu API)
def ask_foundry_local(context_text, user_question, model_name="phi-3.5-mini"):
    prompt = f"""
Sen yetenekli bir mutfak asistanısın. Aşağıdaki tarif bilgilerini kullanarak kullanıcının sorusuna cevap ver.
Eğer cevap bu metinlerde yoksa, "Bilmiyorum, sistemimde bu tarif yok" şeklinde dürüstçe yanıtla.

[BAĞLAM (TARİFLER)]:
{context_text}

[KULLANICI SORUSU]:
{user_question}
    """
    
    # Foundry Local, OpenAI uyumlu bir API sunar. (Örn. yerel portta)
    # Eğer Foundry Local SDK'nın native (yerel) metotlarını kullanmak istersen, 
    # ChatClient sınıfının completeChat fonksiyonu da benzer bir işlev görür.
    try:
        # Not: Foundry Local servisini başlattığınız varsayılmaktadır.
        client = OpenAI(base_url="http://localhost:8080/v1", api_key="foundry-local")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Sen yerel RAG asistanısın."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Foundry Local ile iletişim kurulamadı. Hata: {e}\nNot: Foundry Local servisinin çalıştığından emin olun."

def main():
    print("Mutfak Asistanı Başlatılıyor... (Embedding Modeli yükleniyor)")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Hazır!\n")
    
    while True:
        question = input("\nYemek Asistanına Sorunuz ('çıkış' için q): ")
        if question.lower() in ['q', 'çıkış', 'quit']:
            break
            
        print("En iyi tarifler aranıyor...")
        query_embedding = embedder.encode(question).tolist()
        top_results = get_top_recipes(query_embedding)
        
        context_text = ""
        print("\n--- Bulunan Kaynaklar ---")
        for i, (score, title, text) in enumerate(top_results):
            print(f"{i+1}. {title} (Benzerlik: %{int(score*100)})")
            context_text += f"\n---\n{text}\n"
            
        print("\nYapay Zeka Yanıtlıyor (Microsoft Foundry Local üzerinden)...")
        # Foundry Local'da çalıştırdığınız modelin adı (örn: phi-3.5-mini)
        answer = ask_foundry_local(context_text, question, model_name="phi-3.5-mini")
        
        print("\n" + "="*50)
        print(answer)
        print("="*50)

if __name__ == "__main__":
    main()
