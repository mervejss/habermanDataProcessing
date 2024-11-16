import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt


# Veri kümesini yükleme fonksiyonu
def load_haberman_data():
    data = pd.DataFrame({
        "Age": [30, 31, 31, 35, 35, 37, 38, 38, 38, 39, 39, 39, 39, 40, 41, 41, 41, 42, 43, 43],
        "Year": [64, 64, 65, 64, 66, 65, 69, 66, 66, 67, 68, 69, 67, 66, 65, 69, 66, 69, 64, 64],
        "Positive": [1, 1, 0, 1, 0, 0, 3, 1, 0, 3, 3, 1, 2, 0, 0, 2, 1, 2, 2, 3],
        "Survival": ["positive", "positive", "negative", "positive", "negative",
                     "negative", "positive", "negative", "negative", "positive",
                     "positive", "positive", "positive", "negative", "negative",
                     "positive", "positive", "positive", "positive", "positive"]
    })
    return data


# Sürekli verileri eşit genişlik aralıklarına ayırma fonksiyonu
def discretize_features(data, feature, bins, labels):
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    data[f"{feature}_discretized"] = discretizer.fit_transform(data[[feature]]).astype(int)
    return data


# Bilgi kazancı hesaplama fonksiyonu
def compute_information_gain(data, feature, target="Survival"):
    return mutual_info_score(data[target], data[feature])


# İşlem akışı
def process_data():
    # Veriyi yükle
    data = load_haberman_data()

    # Sonuçları görselleştirmek için saklama
    ig_results = {}

    # Nitelikler ve hedef değişken
    features = ["Age", "Year", "Positive"]
    target = "Survival"

    # 3 ve 4 eşit genişlik aralıkları için bilgi kazancı hesaplama
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, bins in enumerate([3, 4]):
        for feature in features:
            data = discretize_features(data, feature, bins, labels=False)
            ig = compute_information_gain(data, f"{feature}_discretized", target)
            ig_results[f"{feature}_bins_{bins}"] = ig

        # Görselleştirme
        axes[i].bar(ig_results.keys(), ig_results.values(), color='skyblue')
        axes[i].set_title(f"Information Gain for {bins} Equal-Width Bins")
        axes[i].set_xlabel("Features")
        axes[i].set_ylabel("Information Gain")
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("information_gain_visualization.png")
    plt.show()

    return ig_results
