import pandas as pd
from statistics_and_visuals import analyze_and_visualize
from normalization_and_histograms import normalize_and_discretize
from information_gain import process_data


# Veri setinin yolunu belirtiyoruz
file_path = r"haberman.dat"

# Veri setini okuyoruz, ancak @data kısmından sonra başlaması için skiprows kullanıyoruz
column_names = ["Age", "Year", "Positive", "Survival"]
data = pd.read_csv(file_path, header=None, names=column_names, skiprows=8)

# Veri setinin ilk 10 satırını yazdırıyoruz. Normalde toplam 306 satır mevcut.
print("Veri setinin ilk 10 satırı:")
print(data.head(10))

# 1. Kısım
data = pd.read_csv(file_path, header=None, names=column_names, skiprows=8)
# Analiz ve görselleştirme fonksiyonunu çağırma
analyze_and_visualize(data)



# 2. Kısım

# Normalizasyon ve ayrıklaştırma işlemlerini çağırma
normalize_and_discretize(data, n_bins=5)


# 3. Kısım
print("Information Gain Results:", process_data())