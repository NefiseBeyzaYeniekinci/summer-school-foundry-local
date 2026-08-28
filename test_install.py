import sys

try:
    # Önce genel paketi kontrol edelim
    import foundry_local_sdk # veya openai (foundry local sdk openai uyumlu API kullanır genellikle)
    print("Foundry Local SDK yüklendi!")
except ImportError:
    print("Henüz kütüphane dahil edilemedi veya isim farklı.")

def main():
    print("--- Microsoft Foundry Local Projesi ---")
    print("Ortam başarıyla kuruldu.")

if __name__ == "__main__":
    main()
