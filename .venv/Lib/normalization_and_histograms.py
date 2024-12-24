import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalize_and_discretize(data):
    """
    Haberman veri kümesi üzerinde Min-Max ve Z-Score normalizasyon işlemlerini gerçekleştirir.
    Histogramlarda ortalama, medyan ve yoğunluk eğrisi gibi ayrıntılı görselleştirme özellikleri sunar.

    Args:
        data (pd.DataFrame): Normalizasyon işlemleri için kullanılan veri kümesi.
    """
    # Sürekli nitelikler
    # Veri kümesindeki sayısal (sürekli) sütunların bir listesi oluşturulur.
    # Bu sütunlar üzerinde normalizasyon işlemleri yapılacaktır.
    numeric_columns = ["Age", "Year", "Positive"]

    # Min-Max ve Z-Score Normalizasyonu için hazırlık
    # MinMaxScaler, verileri [0,1] aralığına dönüştürmek için kullanılır.
    # StandardScaler ise verileri ortalaması 0 ve standart sapması 1 olan bir dağılıma dönüştürür.
    minmax_scaler = MinMaxScaler()
    zscore_scaler = StandardScaler()

    # Min-Max Normalizasyonu
    # MinMaxScaler ile normalizasyon işlemi yapılır ve sonuçlar yeni bir DataFrame'e aktarılır.
    minmax_normalized = pd.DataFrame(
        minmax_scaler.fit_transform(data[numeric_columns]),
        columns=numeric_columns
    )

    # Z-Score Normalizasyonu
    # StandardScaler ile normalizasyon işlemi yapılır ve sonuçlar yeni bir DataFrame'e aktarılır.
    zscore_normalized = pd.DataFrame(
        zscore_scaler.fit_transform(data[numeric_columns]),
        columns=numeric_columns
    )

    # Normalizasyon Sonuçlarını Yazdırma
    # Her iki normalizasyon yönteminin ilk 5 satırı yazdırılarak sonuçların doğruluğu kontrol edilir.
    print("\nMin-Max Normalizasyonu Sonuçları:")
    print(minmax_normalized.head())
    print("---------------------------------------------")

    print("\nZ-Score Normalizasyonu Sonuçları:")
    print(zscore_normalized.head())

    # Histogramları Çizme
    # Her bir sürekli değişken için Min-Max ve Z-Score normalizasyonlarının histogramları çizilir.
    fig, axes = plt.subplots(len(numeric_columns), 2, figsize=(15, 12))
    fig.suptitle("Haberman Veri Kümesi: Normalizasyon Histogramları", fontsize=16)

    for i, col in enumerate(numeric_columns):
        # Min-Max Normalizasyon Histogram
        # Seaborn kütüphanesi kullanılarak histogram çizilir. Yoğunluk eğrisi de eklenir (kde=True).
        sns.histplot(minmax_normalized[col], bins=10, kde=True, ax=axes[i, 0], color='skyblue', edgecolor='black')

        # Ortalama ve Medyan Hesaplama
        # Min-Max normalizasyonu yapılan verilerde ortalama ve medyan hesaplanır.
        mean_minmax = minmax_normalized[col].mean()
        median_minmax = minmax_normalized[col].median()

        # Ortalama ve Medyan Çizgileri
        # Histogram üzerine kırmızı çizgiyle ortalama, yeşil çizgiyle medyan eklenir.
        axes[i, 0].axvline(mean_minmax, color='red', linestyle='--', linewidth=1.5, label='Mean')
        axes[i, 0].axvline(median_minmax, color='green', linestyle='-', linewidth=1.5, label='Median')
        axes[i, 0].set_title(f"{col} - Min-Max Normalization")
        axes[i, 0].set_xlabel("Değerler")
        axes[i, 0].set_ylabel("Frekans")
        axes[i, 0].legend()

        # Z-Score Normalizasyon Histogram
        # Seaborn kullanılarak Z-Score normalizasyonu için histogram ve yoğunluk eğrisi çizilir.
        sns.histplot(zscore_normalized[col], bins=10, kde=True, ax=axes[i, 1], color='lightgreen', edgecolor='black')

        # Ortalama ve Medyan Hesaplama
        # Z-Score normalizasyonu yapılan verilerde ortalama ve medyan hesaplanır.
        mean_zscore = zscore_normalized[col].mean()
        median_zscore = zscore_normalized[col].median()

        # Ortalama ve Medyan Çizgileri
        # Histogram üzerine kırmızı çizgiyle ortalama, yeşil çizgiyle medyan eklenir.
        axes[i, 1].axvline(mean_zscore, color='red', linestyle='--', linewidth=1.5, label='Mean')
        axes[i, 1].axvline(median_zscore, color='green', linestyle='-', linewidth=1.5, label='Median')
        axes[i, 1].set_title(f"{col} - Z-Score Normalization")
        axes[i, 1].set_xlabel("Değerler")
        axes[i, 1].set_ylabel("Frekans")
        axes[i, 1].legend()

        # İstatistiksel Bilgiler
        # Histogramların üzerine ortalama ve medyan bilgileri metin kutusu içinde yazdırılır.
        axes[i, 0].text(0.95, 0.95, f"Mean: {mean_minmax:.2f}\nMedian: {median_minmax:.2f}",
                        transform=axes[i, 0].transAxes, fontsize=9, verticalalignment='top',
                        horizontalalignment='right', bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
        axes[i, 1].text(0.95, 0.95, f"Mean: {mean_zscore:.2f}\nMedian: {median_zscore:.2f}",
                        transform=axes[i, 1].transAxes, fontsize=9, verticalalignment='top',
                        horizontalalignment='right', bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

    # Tüm grafik düzenini optimize etme
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
