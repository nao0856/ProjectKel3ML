import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Memuat Data
df_crime = pd.read_csv("Presentase Penyelesaian Tindak Pidana di Indonesia tahun 2021-2022.csv")

# Hapus baris 'INDONESIA' yang merupakan total, karena bukan data per daerah
df_crime = df_crime[df_crime['Kepolisian Daerah'] != 'INDONESIA']

# 2. Data Wrangling dan Feature Engineering (Sama seperti di notebook)
# Hitung rata-rata persentase penyelesaian
df_crime['Rata_Rata_Penyelesaian(%)'] = (df_crime['Penyelesaian tindak pidana 2021(%)'] + df_crime['Penyelesaian tindak pidana 2022(%)']) / 2

# Lihat distribusi data untuk menentukan ambang batas yang sesuai.
print("Deskripsi Rata-Rata Persentase Penyelesaian:")
print(df_crime['Rata_Rata_Penyelesaian(%)'].describe())

# Tentukan ambang batas (thresholds) berdasarkan kuartil atau nilai lain yang relevan.
# Contoh: Rendah < 55%, Sedang 55%-70%, Tinggi > 70%
batas_rendah = 55
batas_tinggi = 70

# Buat fungsi untuk mengklasifikasikan tingkat penanganan.
def klasifikasi_tingkat_penanganan(persentase):
    if persentase > batas_tinggi:
        return 'Tinggi'
    elif persentase >= batas_rendah and persentase <= batas_tinggi:
        return 'Sedang'
    else:
        return 'Rendah'

# Buat variabel target kategoris 'Tingkat_Penanganan'.
df_crime['Tingkat_Penanganan'] = df_crime['Rata_Rata_Penyelesaian(%)'].apply(klasifikasi_tingkat_penanganan)

# Pilih fitur (X) dan variabel target (y).
# Untuk kasus ini, kita akan mencoba memprediksi tingkat penanganan berdasarkan jumlah tindak pidana.
X = df_crime[['Rata_Rata_Penyelesaian(%)']]
y = df_crime['Tingkat_Penanganan']

# 4. Latih dan Simpan Scaler
# Inisialisasi dan latih StandardScaler pada seluruh data fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simpan scaler ke file
joblib.dump(scaler, 'scaler.joblib')
print("Scaler berhasil dilatih dan disimpan sebagai 'scaler.joblib'")

# 5. Latih dan Simpan Model
# Inisialisasi model Logistic Regression
model = LogisticRegression(random_state=42)

# Latih model pada seluruh data yang sudah di-scaling
model.fit(X_scaled, y)

# Simpan model ke file
joblib.dump(model, 'model.joblib')
print("Model berhasil dilatih dan disimpan sebagai 'model.joblib'")
