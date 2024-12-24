import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_and_visualize(data):
    """
    Haberman veri setini analiz eder ve görselleştirir.
    """

    # Sayısal özellikler belirleniyor (Yaş, Yıl ve Pozitif nod sayısı)
    numeric_columns = ["Age", "Year", "Positive"]

    # İstatistiksel özetler için bir sözlük oluşturuluyor
    summary_stats = {}

    # Her bir sütun için beş sayı özeti ve diğer istatistikleri hesapla
    for col in numeric_columns:
        # Beş sayı özeti (min, Q1, median, Q3, max)
        five_num_summary = np.percentile(data[col], [0, 25, 50, 75, 100])

        # Ortalama (mean)
        mean = data[col].mean()

        # Mod (en sık tekrar eden değer), mod birden fazla olabilir. Bu nedenle ilk değeri alıyoruz.
        mode = data[col].mode().iloc[0]

        # Çeyrekler arası genişlik (InterQuartile Range - IQR)
        iqr = five_num_summary[3] - five_num_summary[1]

        # Varyans (variance)
        variance = data[col].var()

        # Standart sapma (standard deviation)
        std_dev = data[col].std()

        # Aykırı değer kontrolü (IQR yöntemi ile)
        # Alt ve üst sınırlar Q1 - 1.5 * IQR ve Q3 + 1.5 * IQR olarak hesaplanır.
        lower_bound = five_num_summary[1] - 1.5 * iqr
        upper_bound = five_num_summary[3] + 1.5 * iqr

        # Aykırı değerler (alt veya üst sınırların dışında kalan değerler)
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col].values

        # İstatistikleri saklama
        summary_stats[col] = {
            "Min": five_num_summary[0],
            "Q1": five_num_summary[1],
            "Median": five_num_summary[2],
            "Q3": five_num_summary[3],
            "Max": five_num_summary[4],
            "Mean": mean,
            "Mode": mode,
            "IQR": iqr,
            "Variance": variance,
            "Standard Deviation": std_dev,
            "Outliers": outliers.tolist()  # Aykırı değerleri liste olarak saklıyoruz
        }

    # Hesaplanan istatistikleri yazdırma
    print("İstatistiksel Özet:")
    for col, stats in summary_stats.items():
        print(f"\n{col}:")
        for stat, value in stats.items():
            print(f"  {stat}: {value}")

    # Görselleştirme: Kutu diyagramları (boxplot)
    # Kutu diyagramları her bir sayısal sütunun değerlerini ve aykırı değerleri görselleştirmek için kullanılır.
    fig, axes = plt.subplots(1, len(numeric_columns), figsize=(15, 5))
    fig.suptitle("Haberman Veri Kümesi: Kutu Diyagramları", fontsize=16)

    for ax, col in zip(axes, numeric_columns):
        ax.boxplot(data[col])  # Kutu diyagramı çizimi
        ax.set_title(f"{col}")  # Her bir sütun için başlık
        ax.set_ylabel("Değerler")  # Y ekseni etiketi

    # Görselleştirme düzenlemesi
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
