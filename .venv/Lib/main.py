import pandas as pd
#from information_gain import process_data

"""
248229001001
Merve Başak DEMİRTAŞ
Bilgisayar Mühendisliği A.B.D. / Bilgisayar Mühendisliği Yüksek Lisans / 1. Sınıf
Haberman Veri Seti / Veri Ön İşleme Teknikleri Dersi Uygulaması

3 tane fiel var benim veri setimde. 3 fieldi da AYRIK OLMAYAN (SÜREKLİ, CONTINUOUS) olarak kabul ettim 
ve ödevin gerekli aşamalarında her birini equalwith yöntemi ile n eşit parçaya bölerek sürekli hale getirdim sonrasında işlemleri gerçekleştirdim.
"""

### 1. BÖLÜM: Veri Setinin Yüklenmesi ve Genel İnceleme

# Veri setinin yolunu belirtiyoruz
file_path = r"haberman.dat"

# Veri setini okuyoruz, ancak @data kısmından sonra başlaması için skiprows kullanıyoruz
# Burada skiprows=8 parametresi, veri dosyasındaki ilk 8 satırı atlamak için kullanılır. Bu satırlar genellikle başlıklar veya meta veriler olabilir.
# Ayrıca sütun adlarını verisetindeki gibi "Age", "Year", "Positive" ve "Survival" olarak belirliyoruz.
column_names = ["Age", "Year", "Positive", "Survival"]
data = pd.read_csv(file_path, header=None, names=column_names, skiprows=8)

# Veri setinin ilk 10 satırını yazdırıyoruz. Normalde toplam 306 satır mevcut.
# Bu işlem, veri setinin yapısını görmek ve genel bir inceleme yapmak için kullanılır.
print("Veri setinin ilk 10 satırı:")
print(data.head(10))

print("---------------------------------------------")

# Eksik verileri kontrol etme
# Bu adımda, veri setinde eksik (NaN) değerlerin olup olmadığını kontrol ediyoruz.
# data.isnull().sum() fonksiyonu, her sütundaki eksik değerlerin sayısını verir. Eğer bir sütunda eksik veri varsa, bu değeri doldurmak veya o satırı çıkarmak gerekebilir.
print("Eksik Veri Kontrolü")
print(data.isnull().sum())  # Her sütundaki eksik veri sayısını gösterir
print("---------------------------------------------")

# Hatalı formatları kontrol etme
# Burada, her sütunun veri tiplerini kontrol ediyoruz. Özellikle sayısal verilere (Age, Year, Positive) sayısal tür,
# 'Survival' sütununa ise kategorik veri türü atanıp atanmadığını kontrol ederiz.
# Eğer veri türü yanlışsa, uygun türdeki verilere dönüştürme işlemi yapılabilir.
print("Hatalı Format Kontrolü")
print(data.dtypes)  # Sütunların veri türlerini gösterir
print("---------------------------------------------")

# Temel istatistiksel veriler
# Bu aşamada, sayısal sütunlar için temel istatistiksel bilgiler hesaplanır: ortalama (mean), standart sapma (std), minimum (min),
# maksimum (max) ve çeyrek dilimler (25%, 50%, 75%). Bu bilgiler, veri setinin genel dağılımını anlamaya yardımcı olur.
print("Temel İstatistiksel Veriler")
print(data.describe())  # Sayısal verilerin temel istatistiksel özetini gösterir
print("---------------------------------------------")

# Kategorik verilerin kontrolü
# 'Survival' sütunundaki benzersiz değerleri ve bu değerlerin kaç kez tekrarlandığını kontrol ediyoruz.
# Bu işlem, verinin dengesiz olup olmadığını anlamaya yardımcı olur. Burada 'negative' ve 'positive' olmak üzere iki kategori bulunmaktadır.
print("Kategorik Verilerin Kontrolü")
print(data['Survival'].value_counts())  # 'Survival' sütunundaki kategorik değerlerin sayısını gösterir
print("---------------------------------------------")


print("---------------------------------------------")

### 2. BÖLÜM: İstatistiksel analiz ve görselleştirme
"""
ÖDEV SORUSU :
1. X veri kümesinin e-postada belirtilen her bir niteliği (attribute) için beş sayı özeti (five
number summary), ortalama (mean), mod (mode), çeyrekler arası genişlik (InterQuartile
Range), varyans (variance), standart sapma (standard deviation) ve varsa aykırı (outlier)
değerlerini hesaplayınız. Her bir nitelik için kutu diyagramı (boxplot) çizimini
gerçekleştiriniz.
"""
# Burada analyze_and_visualize fonksiyonunu çağırarak veri setinin her bir sayısal niteliği için istatistiksel analiz yapıyoruz.
# Bu analizde beş sayı özeti, ortalama, mod, IQR, varyans, standart sapma ve aykırı değerler hesaplanır.
# Ayrıca, veri dağılımını görselleştirmek için her bir nitelik için kutu diyagramı (boxplot) oluşturulur.

# statistics_and_visuals modülünden analyze_and_visualize fonksiyonunu çağırıyoruz
from statistics_and_visuals import analyze_and_visualize

# analyze_and_visualize fonksiyonunu çağırarak veriyi analiz ediyoruz
analyze_and_visualize(data)


print("---------------------------------------------")
print("---------------------------------------------")

### 3. BÖLÜM A) : Normalizasyon ve Histogramlar
"""
ÖDEV SORUSU :
2. X veri kümesi için aşağıdaki işlemleri gerçekleştiriniz.

a) Veri kümesinin e-postada belirtilen her bir niteliği için ayrı ayrı 
[0-1] aralığına minmax normalizasyon ve z-score normalizasyon değerlerini hesaplayınız.
"""

# normalization_and_histograms modülünden normalize_and_discretize fonksiyonunu çağırıyoruz
from normalization_and_histograms import normalize_and_discretize

# normalize_and_discretize fonksiyonunu çağırarak veri normalizasyonu yapıp histogram çizdiriyoruz.
normalize_and_discretize(data)


print("---------------------------------------------")
print("---------------------------------------------")

### 3. BÖLÜM B) : Ayrıklaştırma ve Histogramlar
"""
ÖDEV SORUSU :
2. X veri kümesi için aşağıdaki işlemleri gerçekleştiriniz.
b) Ayrık olmayan nitelik değerlerini (continuous attributes) n eşit genişlik (n equalwidth) bölme yöntemleri ile ayrık hale dönüştürünüz (n değerini kendiniz belirleyebilirsiniz).
Her bir yöntem sonucu elde edilen verilerin her bir niteliği için frekans tabanlı histogram
grafiklerini çiziniz. Veri kümenizde belirtilen tüm nitelikler ayrık (discrete) ise bu seçeneği
ihmal ediniz. Sadece sürekli değerli nitelikler için gerçekleştirilecektir.
"""

# discretization_and_histograms modülünden discretize_and_plot fonksiyonunu çağırıyoruz
from discretization_and_histograms import discretize_and_plot
# discretize_and_plot fonksiyonunu çağırarak Ayrıklaştırma ve histogram çizim işlemlerini yapıyoruz.
discretize_and_plot(data, n_bins=5)




print("---------------------------------------------")
print("---------------------------------------------")
### 4. BÖLÜM : Ayrıklaştırma ve Information Gain Hesaplama
"""
3. X veri kümesi için e-postada belirtilen her bir niteliğin bilgi kazancı (information gain)
değeri hesaplanacaktır. Sürekli değerli (Ayrık olmayan, Continuous) olan nitelikler varsa
aşağıdaki işlemleri gerçekleştiriniz. Veri kümenizin belirtilen niteliklerinin tamamı ayrık
(discrete) ise a ve b seçeneklerini ihmal ediniz ve direkt olarak bilgi kazancı değerlerini
hesaplayınız.
"""

### 4. BÖLÜM A) : Equal-Width=3  Ayrıklaştırma ve Information Gain Hesaplama
"""
a) Veri kümesinde ayrık olmayan nitelik değerlerini 3 eşit genişlik (3 equal-width)
değeri olacak şekilde parçalara ayırdıktan sonra her bir nitelik için bilgi kazancını (information
gain) hesaplayınız.
"""

# discretization_and_histograms modülünden discretize_and_plot fonksiyonunu çağırıyoruz
from equal_width_discretization_3bins  import discretize_and_calculate_info_gain
# discretize_and_calculate_info_gain fonksiyonunu çağırarak Ayrıklaştırma ve histogram çizim işlemlerini yapıyoruz.
discretize_and_calculate_info_gain(data)

print("---------------------------------------------")
print("---------------------------------------------")

### 4. BÖLÜM B) : Equal-Width=4  Ayrıklaştırma ve Information Gain Hesaplama
"""
b) Veri kümesinde ayrık olmayan nitelik değerlerini 4 eşit genişlik (4 equal-width)
değeri olacak şekilde parçalara ayırdıktan sonra her bir nitelik için bilgi kazancını (information
gain) hesaplayınız.
"""

# discretization_and_histograms modülünden discretize_and_plot fonksiyonunu çağırıyoruz
from equal_width_discretization_4bins  import discretize_and_calculate_info_gain
# discretize_and_calculate_info_gain fonksiyonunu çağırarak Ayrıklaştırma ve histogram çizim işlemlerini yapıyoruz.
discretize_and_calculate_info_gain(data)