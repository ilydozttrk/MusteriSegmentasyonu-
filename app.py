import streamlit as st
import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
import plotly.express as px 
import os
import time

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Müşteri Segmentasyon Projesi", layout="wide", page_icon="🛍️")

# --- 1. VERİYİ YÜKLEME ---
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("HATA: 'online_retail_II.xlsx' dosyası bulunamadı! Lütfen dosyayı proje klasörüne ekleyin.")
    st.stop()

# --- KENAR ÇUBUĞU (SIDEBAR) ---
with st.sidebar:
    st.title("⚙️ Kontrol Paneli")
    st.markdown("---")
    
    st.subheader("Algoritma Ayarları")
    k_value = st.slider("Küme Sayısı (K)", min_value=3, max_value=6, value=3, help="Müşterileri kaç gruba ayırmak istediğinizi seçin.")
    
    st.markdown("---")
    
    st.subheader("📁 Veri Seti Bilgisi")
    st.info(f"**Toplam Satır:** {df.shape[0]:,}\n\n**Analiz Yılı:** 2010-2011")
    
    st.markdown("---")
    
    st.markdown("### 🎓 Geliştirici")
    st.markdown("**İlayda ÖZTÜRK**")
    st.caption("Bilgisayar Mühendisliği | 3. Sınıf")
    st.caption("Örüntü Tanıma Dersi Projesi © 2025")

# --- ANA BAŞLIK ---
st.title("🛍️ Müşteri Segmentasyon Analizi")
st.markdown(f"**Sistem Durumu:** Hazır 🟢 | **Seçilen Segmentasyon:** {k_value} Küme")

# --- 2. VERİ İŞLEME VE AKILLI İSİMLENDİRME ---
def process_data(df, n_clusters):
    df_clean = df.copy()
    df_clean.dropna(subset=["Customer ID"], inplace=True)
    df_clean = df_clean[~df_clean["Invoice"].astype(str).str.contains("C", na=False)]
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["Price"] > 0)]
    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["Price"]
    df_clean["Customer ID"] = df_clean["Customer ID"].astype(int)
    
    analiz_tarihi = df_clean["InvoiceDate"].max() + dt.timedelta(days=2)
    rfm = df_clean.groupby('Customer ID').agg({
        'InvoiceDate': lambda date: (analiz_tarihi - date.max()).days,
        'Invoice': lambda num: num.nunique(),
        'TotalPrice': lambda price: price.sum()
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm["Monetary"] > 0]
    
    if os.path.exists("yeni_musteriler.csv"):
        try:
            yeni_veri = pd.read_csv("yeni_musteriler.csv", index_col=0)
            rfm = pd.concat([rfm, yeni_veri])
        except:
            pass

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(rfm_scaled)
    rfm['Cluster_ID'] = clusters
    
    cluster_means = rfm.groupby('Cluster_ID')['Monetary'].mean().sort_values()
    
    if n_clusters == 3:
        seg_names = ['Riskli/Kayıp', 'Standart', 'VIP (Şampiyon)']
    elif n_clusters == 4:
        seg_names = ['Kayıp', 'Riskli', 'Sadık', 'VIP']
    elif n_clusters == 5:
        seg_names = ['Kayıp', 'Uykuda', 'Potansiyel', 'Sadık', 'Şampiyon']
    elif n_clusters == 6:
        seg_names = ['Kayıp', 'Çok Riskli', 'Riskli', 'Potansiyel', 'Sadık', 'Şampiyon']
    else:
        seg_names = [f"Segment {i}" for i in range(n_clusters)]
        
    mapping = {old_id: name for old_id, name in zip(cluster_means.index, seg_names)}
    rfm['Segment'] = rfm['Cluster_ID'].map(mapping)
    
    pca = PCA(n_components=3)
    pca_components = pca.fit_transform(rfm_scaled)
    rfm['PCA1'] = pca_components[:, 0]
    rfm['PCA2'] = pca_components[:, 1]
    
    rfm.sort_index(inplace=True)
    
    return rfm, scaler, kmeans, mapping

with st.spinner('Veriler analiz ediliyor...'):
    rfm, scaler, kmeans, segment_map = process_data(df, k_value)

st.session_state.rfm_data = rfm.copy()

# --- ARAYÜZ SEKMELERİ ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Özet", 
    "🔍 Müşteri Bul", 
    "🧊 3D Görsel", 
    "📉 PCA Analizi", 
    "➕ Yeni Müşteri Ekle", 
    "📋 Detaylı Rapor"
])

# --- TAB 1: ÖZET ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Müşteri", rfm.shape[0])
    col2.metric("Ortalama Harcama", f"{rfm['Monetary'].mean():.0f} TL")
    col3.metric("En Sık Gelen", f"{int(rfm['Frequency'].max())} Kez")
    col4.metric("Küme Sayısı", k_value)
    
    st.divider()
    st.subheader("Segment Dağılımı")
    
    summary_stats = rfm.groupby('Segment').agg({
        'Cluster_ID': 'count',
        'Monetary': 'mean',
        'Recency': 'mean',
        'Frequency': 'mean'
    }).reset_index()
    summary_stats.columns = ['Segment', 'Kişi Sayısı', 'Ort. Harcama', 'Ort. Recency', 'Ort. Sıklık']
    
    fig_bar = px.bar(summary_stats, x="Segment", y="Kişi Sayısı", color="Segment",
                     title="Gruplardaki Müşteri Sayıları ve Ortalamaları",
                     text="Kişi Sayısı",
                     hover_data={'Segment': False, 'Kişi Sayısı': True, 'Ort. Harcama': ':.2f', 'Ort. Recency': ':.0f', 'Ort. Sıklık': ':.1f'},
                     category_orders={"Segment": sorted(rfm['Segment'].unique())})
    
    max_count = summary_stats['Kişi Sayısı'].max()
    fig_bar.update_layout(yaxis_range=[0, max_count * 1.2]) 
    fig_bar.update_traces(textposition='outside') 
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- YENİ EKLENTİ: SEGMENT SÖZLÜĞÜ ---
    st.divider()
    st.markdown("### 📖 Segment Rehberi (Kısa Açıklamalar)")
    
    # Kümelerin anlamlarını içeren sabit sözlük
    descriptions = {
        "Şampiyon": "En çok harcayan, en sık gelen ve en yeni müşteriler.",
        "VIP": "Çok yüksek harcama yapan sadık kitle.",
        "Sadık": "Düzenli alışveriş yapan, güvenilir müşteriler.",
        "Potansiyel": "Umut vaat eden, harcaması artma eğiliminde olanlar.",
        "Standart": "Ortalama harcama ve sıklığa sahip genel kitle.",
        "Uykuda": "Eskiden gelen ama son zamanlarda görünmeyenler.",
        "Riskli": "Kaybetme riski yüksek, harcaması düşmüş grup.",
        "Çok Riskli": "Neredeyse kaybedilmiş, acil müdahale gerekenler.",
        "Kayıp": "Uzun süredir gelmeyen ve harcaması çok düşük olanlar."
    }
    
    # Ekranda sadece mevcut segmentlerin açıklamasını gösterelim
    current_segments = sorted(rfm['Segment'].unique(), reverse=True) # İyiden kötüye
    
    # Güzel bir tablo oluşturmak için liste hazırlayalım
    desc_data = []
    for seg in current_segments:
        # Segment isminin içinde geçen anahtar kelimeyi bul (Örn: 'VIP (Şampiyon)' içinde 'Şampiyon' var mı?)
        found_desc = "Özel Tanımlı Segment"
        for key, text in descriptions.items():
            if key in seg:
                found_desc = text
                break
        desc_data.append({"Segment Adı": seg, "Açıklama": found_desc})
    
    st.table(pd.DataFrame(desc_data))

# --- TAB 2: MÜŞTERİ BUL (Düzeltilmiş Versiyon) ---
with tab2:
    st.subheader("🆔 ID ile Müşteri Sorgulama")
    
    # Session State'teki en güncel veriden ID'leri alıyoruz
    min_id = int(st.session_state.rfm_data.index.min())
    max_id = int(st.session_state.rfm_data.index.max())
    
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            # DÜZELTME BURADA:
            # Başlığa (Label) gerçek aralığı yazdık ki kullanıcı görsün.
            # Ama max_value'yu 99999 yaptık ki listede olmayan (silinen) kişileri de aratabilsin.
            selected_id = st.number_input(
                f"Müşteri ID Giriniz (Mevcut Aralık: {min_id} - {max_id}):", 
                min_value=1, 
                max_value=99999, # Teknik sınır geniş bırakıldı
                value=min_id,    # Varsayılan değer
                step=1
            )
        with col2:
            st.write("") 
            st.write("") 
            submitted = st.form_submit_button("🔍 Müşteriyi Getir")
    
    if submitted:
        # 1. DURUM: Müşteri Analiz Listesinde (RFM) VAR
        if selected_id in st.session_state.rfm_data.index:
            cust = st.session_state.rfm_data.loc[selected_id]
            segment_name = cust['Segment']
            
            if "VIP" in segment_name or "Şampiyon" in segment_name:
                st.success(f"🏆 **Müşteri Bulundu:** {segment_name}")
            elif "Kayıp" in segment_name or "Riskli" in segment_name:
                st.error(f"⚠️ **Müşteri Bulundu:** {segment_name}")
            else:
                st.info(f"🛒 **Müşteri Bulundu:** {segment_name}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Recency (Gün)", f"{cust['Recency']:.0f}")
            c2.metric("Frequency (Sıklık)", f"{cust['Frequency']:.0f}")
            c3.metric("Monetary (Tutar)", f"{cust['Monetary']:.2f} TL")
            
        # 2. DURUM: Müşteri Analiz Listesinde YOK -> Dedektiflik Başlasın 🕵️‍♂️
        else:
            st.error(f"❌ {selected_id} numaralı müşteri segmentasyon analizinde YER ALMIYOR.")
            
            # Ham veriye gidip bakalım (Excel Dosyasına)
            raw_user = df[df['Customer ID'] == selected_id]
            
            st.divider()
            st.markdown("### 🕵️‍♂️ Neden Yok? (Otomatik Analiz)")
            
            if raw_user.empty:
                st.warning("👉 **Kesin Sebep:** Bu ID numarası yüklenen Excel dosyasında (Ham Veri) hiç bulunmuyor. Numarayı yanlış yazmış olabilirsiniz.")
            
            else:
                total_spend = (raw_user['Quantity'] * raw_user['Price']).sum()
                cancel_count = raw_user['Invoice'].astype(str).str.contains('C', na=False).sum()
                
                if total_spend <= 0:
                    st.warning(f"👉 **Kesin Sebep (Negatif Bakiye):** Bu müşteri sistemde kayıtlı fakat toplam harcaması **{total_spend:.2f} TL**. Yani yaptığı iadeler, harcamalarından fazla veya eşit.")
                
                elif cancel_count > 0 and total_spend < 10:
                    st.warning(f"👉 **Kesin Sebep (İptal İşlemler):** Müşterinin kayıtlarında **{cancel_count} adet iptal (C)** faturası tespit edildi. Geçersiz işlem sayıldığı için elendi.")
                    
                else:
                    st.warning("👉 **Kesin Sebep (Veri Kalitesi):** Müşteri kaydında 'Birim Fiyat' veya 'Miktar' bilgilerinde 0 veya negatif değerler tespit edildiği için temizlik aşamasında silindi.")
                
                with st.expander("Kanıt: Müşterinin Ham Veri Kayıtlarını Gör"):
                    st.dataframe(raw_user.head())

# --- TAB 3: 3D GÖRSELLEŞTİRME ---
with tab3:
    st.subheader("3D Müşteri Uzayı")
    fig_3d = px.scatter_3d(
        rfm, x='Recency', y='Frequency', z='Monetary',
        color='Segment', opacity=0.7, size_max=10,
        hover_data=['Recency', 'Frequency', 'Monetary'],
        category_orders={"Segment": sorted(rfm['Segment'].unique())}
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# --- TAB 4: PCA ANALİZİ ---
with tab4:
    st.subheader("Boyut İndirgeme Analizi (PCA)")
    fig_pca = px.scatter(
        rfm, x="PCA1", y="PCA2", color="Segment",
        title="PCA ile 2 Boyutlu Gösterim",
        category_orders={"Segment": sorted(rfm['Segment'].unique())}
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # --- YENİ EKLENTİ: PCA YORUMU ---
    st.info("""
    **ℹ️ Grafik Analizi:**
    
    Bu grafik, veri setindeki 3 farklı değişkenin (Recency, Frequency, Monetary) matematiksel olarak sıkıştırılarak 2 temel bileşene (PCA1 ve PCA2) indirgenmiş halidir.
    
    * **Ayrışma Başarısı:** Grafikte farklı renkteki noktalar (segmentler) birbirinden ne kadar net ayrışmışsa, kümeleme algoritması o kadar başarılı çalışmış demektir.
    * **Davranışsal Benzerlik:** Birbirine yakın veya iç içe geçmiş noktalar, o müşterilerin satın alma davranışlarının birbirine çok benzediğini gösterir.
    * **Merkezi Dağılım:** Genellikle 'Standart' müşteriler grafiğin merkezinde toplanırken, 'VIP' veya 'Kayıp' gibi uç davranış gösteren gruplar kenarlara doğru yayılır.
    """)

# --- TAB 5: YENİ MÜŞTERİ EKLE ---
with tab5:
    st.subheader("🧪 Yeni Müşteri Ekle")
    
    with st.form("new_customer_form"):
        c1, c2, c3 = st.columns(3)
        new_r = c1.number_input("Recency (Gün)", 1, 365, 30)
        new_f = c2.number_input("Frequency (Adet)", 1, 1000, 5)
        new_m = c3.number_input("Monetary (TL)", 1.0, 100000.0, 1000.0)
        
        submit_new = st.form_submit_button("💾 Kaydet ve Analiz Et")
    
    if submit_new:
        input_data = pd.DataFrame([[new_r, new_f, new_m]], columns=['Recency', 'Frequency', 'Monetary'])
        input_scaled = scaler.transform(input_data)
        
        pred_cluster_id = kmeans.predict(input_scaled)[0]
        pred_segment_name = segment_map[pred_cluster_id]
        
        new_id = int(rfm.index.max()) + 1
        st.toast(f"ID: {new_id} kaydedildi! Sayfa yenileniyor...", icon="🔄")
        
        row_save = pd.DataFrame({'Recency': [new_r], 'Frequency': [new_f], 'Monetary': [new_m]}, index=[new_id])
        file_name = "yeni_musteriler.csv"
        if not os.path.exists(file_name):
            row_save.to_csv(file_name, mode='w', header=True)
        else:
            row_save.to_csv(file_name, mode='a', header=False)
            
        time.sleep(1.5)
        st.rerun()

# --- TAB 6: DETAYLI RAPOR ---
with tab6:
    st.header("📋 Detaylı Segment Analizi ve Raporu")
    
    selected_segment = st.selectbox("Analiz etmek istediğiniz kümeyi seçin:", sorted(rfm['Segment'].unique()))
    
    filtered_df = rfm[rfm['Segment'] == selected_segment].sort_index()
    
    st.markdown(f"### 📊 {selected_segment} Grubu İstatistikleri")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kişi Sayısı", f"{filtered_df.shape[0]} Kişi")
    c2.metric("Ort. Harcama", f"{filtered_df['Monetary'].mean():.2f} TL")
    c3.metric("Ort. Gelme Süresi", f"{filtered_df['Recency'].mean():.0f} Gün")
    c4.metric("Ort. Sıklık", f"{filtered_df['Frequency'].mean():.1f} Kez")
    
    st.divider()
    
    st.markdown("### 💡 Aksiyon Önerileri (Business Insights)")
    
    if "VIP" in selected_segment or "Şampiyon" in selected_segment:
        st.success("""
        **Strateji:** Bu müşteriler şirketin en değerlileridir.
        - 🌟 Özel müşteri temsilcisi atanmalı.
        - 🎁 Yeni ürünlerde erken erişim hakkı verilmeli.
        - 💎 Sadakat programında 'Premium' statüsüne alınmalı.
        """)
    elif "Sadık" in selected_segment or "Potansiyel" in selected_segment:
        st.info("""
        **Strateji:** Sık geliyorlar ama sepet tutarları artırılabilir.
        - 📈 'Bunu alan şunu da aldı' (Cross-sell) önerileri yapılmalı.
        - 💳 Belirli tutar üzerine (Örn: 500 TL üstü) anında indirim verilmeli.
        - 📢 3 al 2 öde kampanyaları sunulmalı.
        """)
    elif "Kayıp" in selected_segment or "Riskli" in selected_segment:
        st.warning("""
        **Strateji:** Müşteriyi kaybetmek üzereyiz, acil müdahale gerekir!
        - 📧 'Sizi Özledik' temalı e-posta atılmalı.
        - 🏷️ %20 - %30 gibi agresif geri kazanım indirimleri tanımlanmalı.
        - ❓ Neden gelmediklerini öğrenmek için anket gönderilebilir.
        """)
    else: 
        st.write("""
        **Strateji:** Standart müşterileri sadık hale getirmeye çalışın.
        - 📱 Düzenli bülten (newsletter) gönderimi.
        - 🎯 Küçük indirimlerle tekrar gelmeleri teşvik edilmeli.
        """)

    st.divider()
    
    st.write(f"**{selected_segment} Müşteri Listesi Önizlemesi:**")
    st.dataframe(filtered_df.head(10))
    
    csv_data = filtered_df.to_csv().encode('utf-8')
    st.download_button(
        label=f"📥 {selected_segment} Listesini İndir (CSV)",
        data=csv_data,
        file_name=f'{selected_segment}_musterileri.csv',
        mime='text/csv',
    )