import streamlit as st
import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import os # Dosya işlemleri için gerekli kütüphane

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Müşteri Segmentasyon Projesi", layout="wide")

st.title("📊 Müşteri Segmentasyonu ve Analizi")
st.markdown("**Bilgisayar Mühendisliği 3. Sınıf Projesi** | Örüntü Tanıma")

# --- 1. VERİYİ YÜKLEME VE İŞLEME ---
@st.cache_data
def load_and_process_data():
    try:
        # Ana Veriyi Yükle (Excel)
        df = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
    except FileNotFoundError:
        return None, None, None, None

    # --- TEMİZLİK ---
    df_clean = df.copy()
    df_clean.dropna(subset=["Customer ID"], inplace=True)
    df_clean = df_clean[~df_clean["Invoice"].astype(str).str.contains("C", na=False)]
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["Price"] > 0)]
    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["Price"]
    df_clean["Customer ID"] = df_clean["Customer ID"].astype(int)
    
    # --- ANA RFM HESABI ---
    analiz_tarihi = df_clean["InvoiceDate"].max() + dt.timedelta(days=2)
    rfm = df_clean.groupby('Customer ID').agg({
        'InvoiceDate': lambda date: (analiz_tarihi - date.max()).days,
        'Invoice': lambda num: num.nunique(),
        'TotalPrice': lambda price: price.sum()
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm["Monetary"] > 0]
    
    # --- KALICI VERİLERİ (CSV) YÜKLEME VE BİRLEŞTİRME ---
    # Eğer daha önce kaydedilmiş yeni müşteriler varsa, onları da yükle
    if os.path.exists("yeni_musteriler.csv"):
        try:
            # CSV'den oku (Index sütunu Customer ID olacak)
            yeni_veri = pd.read_csv("yeni_musteriler.csv", index_col=0)
            # Ana tabloyla birleştir
            rfm = pd.concat([rfm, yeni_veri])
        except Exception as e:
            st.warning(f"Ek veriler yüklenirken bir sorun oluştu: {e}")

    # --- K-MEANS ---
    # Model tüm veriler (eski + yeni) üzerinde eğitilir
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    rfm['Cluster_ID'] = kmeans.fit_predict(rfm_scaled)
    
    return df_clean, rfm, scaler, kmeans

# Veriyi Yükle
with st.spinner('Veri seti ve kayıtlı müşteriler yükleniyor...'):
    data_result = load_and_process_data()

if data_result[0] is None:
    st.error("HATA: 'online_retail_II.xlsx' dosyası bulunamadı! Lütfen Excel dosyasını bu kodun yanına taşıyın.")
else:
    df, rfm_base, scaler, kmeans = data_result
    
    # Session State'e veriyi atıyoruz
    if 'rfm_data' not in st.session_state:
        st.session_state.rfm_data = rfm_base.copy()
    
    rfm = st.session_state.rfm_data

    st.success("✅ Veri tabanı ve kayıtlı müşteriler başarıyla yüklendi!")

    # --- ARAYÜZ SEKMELERİ ---
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Veri Özeti", "🎨 Görselleştirme", "🔍 Müşteri Bul", "➕ Yeni Müşteri Analizi"])

    with tab1:
        st.header("Veri Seti İstatistikleri")
        col1, col2, col3 = st.columns(3)
        col1.metric("Toplam Müşteri", rfm.shape[0])
        col2.metric("Ortalama Harcama", f"{rfm['Monetary'].mean():.2f} TL")
        col3.metric("Analiz Tarihi", "2011-12-11")
        st.dataframe(rfm.head(10), use_container_width=True)

    with tab2:
        st.header("Müşteri Segmentasyon Grafiği")
        col_x, col_y = st.columns(2)
        with col_x:
            x_axis = st.selectbox("X Ekseni", ["Recency", "Frequency", "Monetary"], index=0)
        with col_y:
            y_axis = st.selectbox("Y Ekseni", ["Recency", "Frequency", "Monetary"], index=2)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=rfm, x=x_axis, y=y_axis, hue='Cluster_ID', palette='viridis', s=100, alpha=0.8, ax=ax)
        plt.title(f"{x_axis} vs {y_axis}")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab3:
        st.header("Müşteri ID ile Sorgulama")
        min_id = int(rfm.index.min())
        max_id = int(rfm.index.max())
        selected_id = st.number_input(f"Müşteri ID ({min_id} - {max_id} arası):", min_value=min_id, max_value=max_id, step=1)
        
        if st.button("Müşteriyi Getir"):
            if selected_id in rfm.index:
                cust = rfm.loc[selected_id]
                cluster = int(cust['Cluster_ID'])
                if cluster == 2:
                    st.success(f"🏆 **VIP Müşteri (Küme {cluster})**")
                elif cluster == 1:
                    st.warning(f"⚠️ **Riskli / Kayıp Müşteri (Küme {cluster})**")
                else:
                    st.info(f"🛒 **Sadık / Standart Müşteri (Küme {cluster})**")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Recency", f"{cust['Recency']:.0f}")
                c2.metric("Frequency", f"{cust['Frequency']:.0f}")
                c3.metric("Monetary", f"{cust['Monetary']:.2f}")
            else:
                st.error("❌ Bu ID veride bulunamadı!")

    with tab4:
        st.header("🧪 Canlı Müşteri Analizi ve Kaydetme")
        st.info("ℹ️ Burada eklediğiniz müşteriler **'yeni_musteriler.csv'** dosyasına kaydedildi.")
        
        c1, c2, c3 = st.columns(3)
        new_recency = c1.number_input("Recency (En son kaç gün önce alışveriş yaptı?)", min_value=1, value=30)
        new_frequency = c2.number_input("Frequency (Kaç defa alışveriş yaptı?)", min_value=1, value=5)
        new_monetary = c3.number_input("Monetary (Toplam Harcama (TL)", min_value=1.0, value=1000.0)
        
        next_id = int(rfm.index.max()) + 1
        new_id = st.number_input("Atanacak ID:", min_value=1, value=next_id)

        if st.button("Analiz Et ve Kalıcı Olarak Kaydet 💾"):
            # Model tahmini
            new_data = pd.DataFrame([[new_recency, new_frequency, new_monetary]], columns=['Recency', 'Frequency', 'Monetary'])
            new_data_scaled = scaler.transform(new_data)
            prediction = kmeans.predict(new_data_scaled)[0]
            
            # Sonuç bildirimi
            st.divider()
            if prediction == 2:
                st.success(f"Sonuç: **Şampiyonlar (VIP) Grubu! (Küme {prediction})**")
            elif prediction == 1:
                st.error(f"Sonuç: **Riskli Grup. (Küme {prediction})**")
            else:
                st.info(f"Sonuç: **Standart Grup. (Küme {prediction})**")
            
            # Yeni satırı oluştur (Sadece ham veriler, Cluster_ID'siz)
            # Çünkü CSV'ye ham veriyi kaydedeceğiz, Cluster ID her açılışta yeniden hesaplanacak
            # Ancak görselde anlık görmek için Cluster ID'yi de ekliyoruz
            
            # 1. Hafızaya Ekleme (Session State)
            row_for_ram = pd.DataFrame({
                'Recency': [new_recency],
                'Frequency': [new_frequency],
                'Monetary': [new_monetary],
                'Cluster_ID': [prediction]
            }, index=[new_id])
            
            st.session_state.rfm_data = pd.concat([st.session_state.rfm_data, row_for_ram])
            rfm = st.session_state.rfm_data # Tabloyu güncelle
            
            # 2. Hard Diske Kaydetme (CSV)
            # Cluster ID olmadan kaydediyoruz ki program açılınca model tekrar hesaplasın
            row_for_csv = pd.DataFrame({
                'Recency': [new_recency],
                'Frequency': [new_frequency],
                'Monetary': [new_monetary]
            }, index=[new_id])
            
            file_name = "yeni_musteriler.csv"
            
            if not os.path.exists(file_name):
                # Dosya yoksa başlıklarla beraber oluştur
                row_for_csv.to_csv(file_name, mode='w', header=True)
            else:
                # Dosya varsa altına ekle (append) ve başlık yazma
                row_for_csv.to_csv(file_name, mode='a', header=False)
            
            st.toast(f"Müşteri ID: {new_id} başarıyla diske kaydedildi!", icon="✅")
            
        # --- SON EKLENENLER TABLOSU ---
        st.divider()
        st.subheader("📋 Sisteme Son Eklenen 10 Müşteri")
        st.dataframe(st.session_state.rfm_data.iloc[::-1].head(10), use_container_width=True)