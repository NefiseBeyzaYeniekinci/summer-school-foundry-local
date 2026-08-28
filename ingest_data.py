import sqlite3
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import os

def init_db(db_path="recipes.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Tabloyu oluşturalım
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            ingredients TEXT,
            directions TEXT,
            site TEXT,
            chunk_text TEXT,
            embedding TEXT
        )
    ''')
    conn.commit()
    return conn

def main():
    print("Vektör modeli yükleniyor... (all-MiniLM-L6-v2)")
    # Türkçe ve İngilizce için küçük ve hızlı bir model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    csv_path = "data/recipes_sample.csv"
    if not os.path.exists(csv_path):
        print(f"Hata: {csv_path} bulunamadı!")
        return
        
    print("Örnek veriler okunuyor...")
    df = pd.read_csv(csv_path)
    
    conn = init_db()
    cursor = conn.cursor()
    
    print("Tarifler vektörlere dönüştürülüp veritabanına kaydediliyor...")
    
    # Tüm tabloyu temizleyelim (tekrar çalıştırılırsa diye)
    cursor.execute('DELETE FROM recipes')
    
    count = 0
    for index, row in df.iterrows():
        title = str(row.get('title', ''))
        ingredients = str(row.get('ingredients', ''))
        directions = str(row.get('directions', ''))
        site = str(row.get('site', ''))
        
        # Yapay zekaya vereceğimiz ve arama yapacağımız metin bloğu (Chunk)
        chunk_text = f"Tarif Adı: {title}\nMalzemeler: {ingredients}\nTarif: {directions}"
        
        # Metni vektöre (sayısal diziye) çevir
        embedding_vector = model.encode(chunk_text).tolist()
        
        # Vektörü JSON formatında metin olarak kaydedelim (SQLite'da liste saklanamadığı için)
        embedding_json = json.dumps(embedding_vector)
        
        cursor.execute('''
            INSERT INTO recipes (title, ingredients, directions, site, chunk_text, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (title, ingredients, directions, site, chunk_text, embedding_json))
        
        count += 1
        if count % 50 == 0:
            print(f"{count}/{len(df)} tarif işlendi...")
            conn.commit()
            
    conn.commit()
    conn.close()
    print(f"\nBaşarılı! Toplam {count} tarif vektörleştirildi ve SQLite veritabanına ('recipes.db') kaydedildi.")

if __name__ == "__main__":
    main()
