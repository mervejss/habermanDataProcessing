import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalize_and_discretize(data, n_bins=5):
    """
    Haberman veri kümesi üzerinde normalizasyon ve ayrıklaştırma işlemleri gerçekleştirir.
    """
    # Sürekli nitelikler
    numeric_columns = ["Age", "Year", "Positive"]

    # Min-Max ve Z-Score Normalizasyonu
    minmax_scaler = MinMaxScaler()
    zscore_scaler = StandardScaler()

    minmax_normalized = pd.DataFrame(minmax_scaler.fit_transform(data[numeric_columns]), columns=numeric_columns)
    zscore_normalized = pd.DataFrame(zscore_scaler.fit_transform(data[numeric_columns]), columns=numeric_columns)

    print("\nMin-Max Normalizasyonu Sonuçları:")
    print(minmax_normalized.head())

    print("\nZ-Score Normalizasyonu Sonuçları:")
    print(zscore_normalized.head())

    # Ayrıklaştırma (n eşit genişlik bölme yöntemi)
    discretized_data = {}
    for col in numeric_columns:
        bins = pd.cut(data[col], bins=n_bins, labels=False)
        discretized_data[col] = bins
        print(f"\n{col} için ayrıklaştırılmış değerler (n={n_bins}):")
        print(bins.value_counts())

    # Histogramları çizme
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Haberman Veri Kümesi: Normalizasyon ve Ayrıklaştırma Histogramları", fontsize=16)

    for i, col in enumerate(numeric_columns):
        # Min-Max Histogram
        axes[i, 0].hist(minmax_normalized[col], bins=10, color='skyblue', edgecolor='black')
        axes[i, 0].set_title(f"{col} - Min-Max Normalization")

        # Z-Score Histogram
        axes[i, 1].hist(zscore_normalized[col], bins=10, color='lightgreen', edgecolor='black')
        axes[i, 1].set_title(f"{col} - Z-Score Normalization")

        # Discretization Histogram
        axes[i, 2].hist(discretized_data[col], bins=n_bins, color='salmon', edgecolor='black')
        axes[i, 2].set_title(f"{col} - Discretization (n={n_bins})")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
