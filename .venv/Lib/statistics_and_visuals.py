import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_and_visualize(data):
    """
    Haberman veri setini analiz eder ve görselleştirir.
    """
    # Özellik isimleri
    numeric_columns = ["Age", "Year", "Positive"]

    # İstatistiksel özetler için bir DataFrame oluşturma
    summary_stats = {}
    for col in numeric_columns:
        # Beş sayı özeti
        five_num_summary = np.percentile(data[col], [0, 25, 50, 75, 100])

        # Ortalamalar ve varyanslar
        mean = data[col].mean()
        mode = data[col].mode().iloc[0]
        iqr = five_num_summary[3] - five_num_summary[1]
        variance = data[col].var()
        std_dev = data[col].std()

        # Aykırı değer kontrolü (IQR yöntemiyle)
        lower_bound = five_num_summary[1] - 1.5 * iqr
        upper_bound = five_num_summary[3] + 1.5 * iqr
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
            "Outliers": outliers.tolist()
        }

    # İstatistikleri yazdırma
    print("İstatistiksel Özet:")
    for col, stats in summary_stats.items():
        print(f"\n{col}:")
        for stat, value in stats.items():
            print(f"  {stat}: {value}")

    # Görselleştirme
    fig, axes = plt.subplots(1, len(numeric_columns), figsize=(15, 5))
    fig.suptitle("Haberman Veri Kümesi: Kutu Diyagramları", fontsize=16)

    for ax, col in zip(axes, numeric_columns):
        ax.boxplot(data[col])
        ax.set_title(f"{col}")
        ax.set_ylabel("Değerler")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
