# 🛍️ E-Ticaret Müşteri Yorum Analiz Sistemi

Bu proje, kadın giyim e-ticaret yorumlarını analiz ederek müşterinin memnuniyet durumunu (Pozitif/Negatif) tahmin eden bir **Doğal Dil İşleme (NLP)** uygulamasıdır. Proje kapsamında **DistilBERT** transformer modeli kullanılmış ve **Streamlit** üzerinden etkileşimli bir panel sunulmuştur.

## 🚀 Proje Hakkında
Bu çalışma, veri temizleme aşamasından model eğitimine ve modelin canlıya alınmasına kadar uçtan uca bir veri bilimi sürecini kapsar.

- **Veri Seti:** Women's E-Commerce Clothing Reviews (Kaggle)
- **Mimari:** DistilBERT-base-uncased (Hugging Face)
- **Arayüz:** Streamlit

## 📊 Eğitim Süreci (Model Training)
Model eğitimi Google Colab ortamında gerçekleştirilmiştir. Eğitim sırasında uygulanan adımlar:
1. **Veri Ön İşleme:** Metinlerdeki gereksiz karakterlerin temizlenmesi ve stopwords'lerin (etkisiz kelimeler) filtrelenmesi.
2. **Sınıflandırma:** 4 ve 5 yıldızlı yorumlar "Pozitif", 1-3 yıldızlı yorumlar "Negatif" olarak etiketlenmiştir.
3. **İnce Ayar (Fine-tuning):** Önceden eğitilmiş DistilBERT modeli, kadın giyim dili özelinde optimize edilmiştir.

> Eğitim kodlarına projedeki `Egitim_Duygu_Analizi.ipynb` dosyasından ulaşabilirsiniz.


## 🧪 Teknolojiler
Python

PyTorch & Transformers (Model yönetimi)

Streamlit (Web arayüzü)

NLTK (Metin temizleme)

Pandas & Matplotlib (Veri analizi ve görselleştirme)

## 🛠️ Kurulum
Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1. Depoyu klonlayın:
   ```bash
   git clone [https://github.com/Mervenuryalcinn/CustomerReviewAnalysis.git](https://github.com/Mervenuryalcinn/CustomerReviewAnalysis.git)
   cd CustomerReviewAnalysis

 2.Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt

 3.Uygulamayı başlatın:
    ```bash
    streamlit run proje.py

  
