import kagglehub
import pandas as pd
import os

def main():
    print("Kaggle'dan veri klasörü indiriliyor...")
    
    try:
        # Sadece klasörü indiriyoruz. (Zaten inmişse anında döner)
        dataset_path = kagglehub.dataset_download("wilmerarltstrmberg/recipe-dataset-over-2m")
        print(f"Veri klasörü: {dataset_path}")
        
        # Klasördeki dosyaları bulalım
        files = os.listdir(dataset_path)
        print("Klasördeki dosyalar:", files)
        
        # İlk csv veya json dosyasını bulalım
        data_file = None
        for f in files:
            if f.endswith('.csv'):
                data_file = f
                break
        
        if data_file:
            full_path = os.path.join(dataset_path, data_file)
            print(f"{data_file} dosyası okunuyor...")
            
            # Dev dosya olduğu için sadece ilk birkaç bin satırı veya sample alabiliriz.
            # Pandas ile tamamını okumak 2M satır için çok RAM isteyebilir ama deneyelim:
            df = pd.read_csv(full_path)
            print(f"Veri seti başarıyla yüklendi! Toplam kayıt: {len(df)}")
            
            print("\nRAG projesinin hızlı çalışabilmesi için 500 rastgele tarif seçiliyor...")
            sample_df = df.sample(n=500, random_state=42)
            
            os.makedirs("data", exist_ok=True)
            sample_file = "data/recipes_sample.csv"
            sample_df.to_csv(sample_file, index=False)
            
            print(f"\nÖrnek veri seti '{sample_file}' dosyasına kaydedildi!")
            print("İlk 5 kayıt:")
            print(sample_df.head())
        else:
            print("Klasörde CSV dosyası bulunamadı.")
            
    except Exception as e:
        print("Bir hata oluştu:", e)

if __name__ == "__main__":
    main()
