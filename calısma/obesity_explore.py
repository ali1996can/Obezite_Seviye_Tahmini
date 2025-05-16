import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
theme_bg = """
<style>
    .stApp {
        background: linear-gradient(to right, #FAA4BD, #F49BAB, #DC8BE0);
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: black !important;
    }
</style>
"""
st.markdown(theme_bg, unsafe_allow_html=True)
st.title("\U0001F3E5 Obezite Seviyesi Tahmini")
st.write("""Bu uygulama, bireylerin yaşam tarzı alışkanlıkları ve fiziksel özelliklerine göre obezite seviyesini tahmin eder.""")
df = pd.read_csv("C:/Users/Ali/OneDrive/Masaüstü/calısma/obezite.csv")
le_nobese = LabelEncoder()
le_calc = LabelEncoder()
le_mtrans = LabelEncoder()
le_caec = LabelEncoder()
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["family_history_with_overweight"] = df["family_history_with_overweight"].map({"yes": 1, "no": 0})
df["FAVC"] = df["FAVC"].map({"yes": 1, "no": 0})
df["SMOKE"] = df["SMOKE"].map({"yes": 1, "no": 0})
df["SCC"] = df["SCC"].map({"yes": 1, "no": 0})
df["CALC"] = le_calc.fit_transform(df["CALC"])
df["MTRANS"] = le_mtrans.fit_transform(df["MTRANS"])
df["CAEC"] = le_caec.fit_transform(df["CAEC"])
df["NObeyesdad"] = le_nobese.fit_transform(df["NObeyesdad"])
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.metric("Model Doğruluğu", f"{accuracy*100:.2f}%")
st.sidebar.header("\U0001F50D Kullanıcı Girdisi")
age = st.sidebar.slider("Yaş", 10, 100, 25)
gender = st.sidebar.selectbox("Cinsiyet", ["Erkek", "Kadın"])
height = st.sidebar.slider("Boy (metre)", 1.3, 2.2, 1.7)
weight = st.sidebar.slider("Kilo (kg)", 30, 150, 70)
favc = st.sidebar.selectbox("Yüksek kalorili yiyecek tüketimi", ["Evet", "Hayır"])
fcvc = st.sidebar.slider("Günlük sebze tüketimi (0-3 arası)", 0.0, 3.0, 2.0)
ncp = st.sidebar.slider("Günlük öğün sayısı", 1, 5, 3)
scc = st.sidebar.selectbox("Kalori bilgisi takibi", ["Evet", "Hayır"])
smoke = st.sidebar.selectbox("Sigara kullanımı", ["Evet", "Hayır"])
ch2o = st.sidebar.slider("Günlük su tüketimi (litre)", 0.0, 3.0, 1.5)
fhwo = st.sidebar.selectbox("Ailede obezite geçmişi", ["Evet", "Hayır"])
faf = st.sidebar.slider("Fiziksel aktivite sıklığı", 0.0, 3.0, 1.0)
tue = st.sidebar.slider("Ekran süresi (saat)", 0.0, 2.0, 1.0)
caec = st.sidebar.selectbox("Atıştırmalık tüketim sıklığı", ["Hayır", "Bazen", "Sık sık", "Her zaman"])
calc = st.sidebar.selectbox("Alkol tüketimi", ["Hayır", "Bazen", "Sık sık", "Her zaman"])
mtrans = st.sidebar.selectbox("Ulaşım şekli", ["Otomobil", "Bisiklet", "Motosiklet", "Toplu Taşıma", "Yürüme"])
obesity_info = {
    "Insufficient_Weight": "Vücut kütle indeksi (BMI) düşük olup, sağlıklı kilonun altındadır. Yetersiz beslenme veya sağlık problemleri olabilir.",
    "Normal_Weight": "BMI normal aralıktadır. Dengeli beslenme ve düzenli egzersiz önerilir.",
    "Overweight_Level_I": "Bir miktar fazla kiloludur. Kilo artışının devam etmemesi için dikkatli olunmalıdır.",
    "Overweight_Level_II": "Fazla kiloludur. Sağlık riskleri artmaktadır, yaşam tarzı değişikliği gerekebilir.",
    "Obesity_Type_I": "Obezitenin ilk seviyesidir. Kalp-damar hastalıkları riski artar.",
    "Obesity_Type_II": "İleri derecede obezitedir. Metabolik hastalıklar riski ciddi ölçüde yüksektir.",
    "Obesity_Type_III": "Morbid obezite olarak bilinir. Acil tıbbi müdahale ve takip gerekebilir."}
if st.sidebar.button("\U0001F52E Tahmin Et"):
    try:
        input_dict = {
        "Age": age,
        "Gender": 1 if gender == "Erkek" else 0,
        "Height": height,
        "Weight": weight,
        "FAVC": 1 if favc == "Evet" else 0,
        "FCVC": fcvc,
        "NCP": ncp,
        "SCC": 1 if scc == "Evet" else 0,
        "SMOKE": 1 if smoke == "Evet" else 0,
        "CH2O": ch2o,
        "family_history_with_overweight": 1 if fhwo == "Evet" else 0,
        "FAF": faf,
        "TUE": tue,
        "CAEC": le_caec.transform([caec])[0] if caec in le_caec.classes_ else 0,
        "CALC": le_calc.transform([calc])[0] if calc in le_calc.classes_ else 0,
        "MTRANS": le_mtrans.transform([mtrans])[0] if mtrans in le_mtrans.classes_ else 0
        }
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[X.columns]
        prediction = model.predict(input_df)[0]
        pred_label = le_nobese.inverse_transform([prediction])[0]
        st.markdown(
            f"""
            <div style='padding: 1rem; background-color: #FFD63A; color: black; border-radius: 10px;'>
                <h4>📌 Tahmin Edilen Obezite Seviyesi:</h4>
                <h2 style='color:#00FF99'>{pred_label}</h2>
                <p>{obesity_info.get(pred_label, "Bu seviye hakkında bilgi bulunamadı.")}</p>
            </div>
            """, unsafe_allow_html=True
        )
    except ValueError as e:
        st.error(f"Girdi verileri hatalı: {e}")
st.subheader("📚 Obezite Hakkında Bilgilendirme")
st.markdown("""
### 🧬 Obezite Nedir?
Obezite, vücutta sağlığı tehdit edecek ölçüde aşırı yağ birikimi olarak tanımlanır. Genellikle **vücut kitle indeksi (BMI)** 30'un üzerinde olan bireyler obez kabul edilir.

### 🎯 Obezitenin Nedenleri
- **Yetersiz fiziksel aktivite**
- **Aşırı ve dengesiz beslenme**
- **Genetik faktörler**
- **Psikolojik nedenler (stres, depresyon)**
- **Metabolik hastalıklar**
### 🩺 Obezite ile İlişkili Sağlık Riskleri
- Kalp-damar hastalıkları
- Tip 2 diyabet
- Hipertansiyon
- Uyku apnesi
- Bazı kanser türleri
### 🛠️ Tedavi Yöntemleri
- **Beslenme düzenlemesi:** Diyetisyen kontrolünde kişiye özel planlama
- **Fiziksel aktivite artırımı:** Günlük egzersiz alışkanlığı
- **Davranışsal terapi:** Psikolojik destek
- **İlaç tedavisi:** Gerekli durumlarda doktor kontrolünde
- **Cerrahi müdahaleler:** Morbid obezite durumlarında mide küçültme gibi operasyonlar

⚠️ Obezite, ciddi ancak önlenebilir bir halk sağlığı problemidir. Erken müdahale ve sağlıklı yaşam alışkanlıkları ile kontrol altına alınabilir.
""")

