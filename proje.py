import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Sayfa Yapılandırması
st.set_page_config(page_title="Müşteri Yorum Analiz Sistemi", page_icon="🛍️")

# Modeli Yükleme
@st.cache_resource
def load_model():
    model_path = "kadin_giyim_duygu_modeli" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

# Arayüz
st.title("🛍️ E-Ticaret Duygu Analizi Paneli")
st.markdown("Müşterilerinizin yorumlarını analiz edin ve memnuniyet oranlarını görün.")

# Kullanıcı Girişi
user_input = st.text_area("Analiz edilecek yorumu buraya yazın:", placeholder="Örn: The dress is beautiful but the size is too small.")

if st.button("Analiz Et"):
    if user_input.strip() != "":
        # Tahmin İşlemi
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()

        # Sonuç Gösterimi
        if prediction == 1:
            st.success(f"### Sonuç: POZİTİF 😊")
            st.balloons()
        else:
            st.error(f"### Sonuç: NEGATİF 😞")
        
        st.write(f"**Modelin Eminlik Oranı:** %{confidence*100:.2f}")
        
        # İlerleme çubuğu ile görselleştirme
        st.progress(confidence)
    else:
        st.warning("Lütfen bir metin girin.")

# Yan Panel (Sidebar) Bilgi
st.sidebar.info("Bu sistem DistilBERT modelini kullanarak kadın giyim yorumlarını analiz eder.")