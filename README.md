# 📊 Customer Segmentation & RFM Analysis System

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 📝 Proje Hakkında (Project Overview)

Bu proje, **Bilgisayar Mühendisliği / Örüntü Tanıma (Pattern Recognition)** dersi kapsamında geliştirilmiş, uçtan uca bir makine öğrenmesi uygulamasıdır. 

Projenin temel amacı, e-ticaret verilerini (Online Retail II dataset) analiz ederek müşteri davranışlarını anlamlandırmak ve şirketler için **Kişiselleştirilmiş Pazarlama (Personalized Marketing)** stratejileri oluşturulmasına olanak tanımaktır.

Sistem, müşterileri **RFM (Recency, Frequency, Monetary)** metrikleri üzerinden analiz eder ve **K-Means (Unsupervised Learning)** kümeleme algoritması kullanarak segmentlere ayırır. Geliştirilen **Streamlit** web arayüzü sayesinde, teknik bilgisi olmayan kullanıcılar bile veri analizi yapabilir ve **yapay zeka modelini kullanarak yeni müşteri tahmini** gerçekleştirebilir.

---

## ⚙️ Teknik Mimari ve Metodoloji (Architecture & Methodology)

Proje 4 ana aşamadan oluşmaktadır:

### 1. Veri Ön İşleme (Data Preprocessing)
* Ham veri setindeki eksik değerlerin (Null values) temizlenmesi.
* İade faturalarının (C ile başlayan faturalar) filtrelenmesi.
* Aykırı değerlerin (Outliers) tespiti ve temizlenmesi.
* Excel (.xlsx) verisinin performans artışı için optimize edilmesi.

### 2. RFM Analizi (Feature Engineering)
Her müşteri için aşağıdaki 3 temel öznitelik (feature) matematiksel olarak hesaplanmıştır:
* **Recency (Yenilik):** Müşterinin son alışverişinden bugüne geçen gün sayısı.
* **Frequency (Sıklık):** Müşterinin toplam alışveriş sayısı.
* **Monetary (Parasal Değer):** Müşterinin toplam harcama tutarı.

### 3. Yapay Zeka Modellemesi (K-Means Clustering)
* Veriler, **StandardScaler** kullanılarak ölçeklendirilmiş (Scaling) ve modele uygun hale getirilmiştir.
* **K-Means Algoritması** kullanılarak müşteriler benzer davranışlarına göre 3 ana kümeye ayrılmıştır:
    * 🏆 **Küme 2 (Champions/VIP):** Yüksek harcama ve sık alışveriş yapanlar.
    * 🛒 **Küme 0 (Loyal/Standard):** Düzenli gelen standart müşteriler.
    * ⚠️ **Küme 1 (At Risk):** Uzun süredir gelmeyen ve kaybetme riski olanlar.

### 4. Canlı Tahmin ve Arayüz (Deployment)
* Model, `pickle` edilmek yerine anlık olarak çalıştırılarak dinamik bir yapı kurulmuştur.
* **Streamlit Session State** kullanılarak geçici hafıza yönetimi sağlanmıştır.
* **Persistence (Kalıcılık):** Yeni eklenen veriler `yeni_musteriler.csv` dosyasına kaydedilerek veri kaybı önlenmiştir.

---

## 🚀 Özellikler (Key Features)

* **📊 Dinamik Dashboard:** Veri setinin genel istatistiklerini (Toplam ciro, müşteri sayısı vb.) anlık gösterir.
* **🎨 İnteraktif Görselleştirme:** Kullanıcı, X ve Y eksenlerini (Recency vs Monetary gibi) değiştirerek kümelerin dağılımını grafik üzerinde inceleyebilir.
* **🔍 Müşteri Sorgulama:** ID numarası girilen müşterinin hangi segmentte olduğu ve harcama detayları sorgulanabilir.
* **🤖 Real-Time Prediction (Canlı Tahmin):** Arayüz üzerinden girilen sanal bir müşteri verisi (Örn: "30 gündür gelmedi, 5000 TL harcadı"), eğitilmiş model tarafından anında analiz edilir ve hangi gruba girdiği tahmin edilir.
* **💾 Veri Kalıcılığı:** Sisteme sonradan eklenen müşteriler yerel diskte (CSV) saklanır, program kapatılıp açılsa bile veriler korunur.

---

## 📂 Dosya Yapısı (File Structure)

```text
├── app.py                      # Ana uygulama dosyası (Main App)
├── online_retail_II.xlsx       # Ham Veri Seti (Dataset)
├── yeni_musteriler.csv         # Sonradan eklenen verilerin veritabanı
├── requirements.txt            # Proje bağımlılıkları
└── README.md                   # Proje dökümantasyonu
