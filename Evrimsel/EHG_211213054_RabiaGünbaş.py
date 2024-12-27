import numpy as np

# Parametreler
populasyon_buyuklugu = 20
problem_boyutu = 5
birey_uzunlugu = problem_boyutu * 15  # Her birey 15 bit ile temsil ediliyor
iterasyon_sayisi = 10000  # Maksimum amaç fonksiyonu hesaplama sayısı
caprazlama_olasiligi = 0.7
mutasyon_olasiligi = 0.001

# Rastgele başlangıç popülasyonu (0 ve 1'lerden oluşan)
populasyon = np.random.randint(2, size=(populasyon_buyuklugu, birey_uzunlugu))

def fitness_hesapla(birey):
    """
    Fitness fonksiyonu: Bireyi [-100, 100] aralığına çevirerek değerlendirme yapar.
    Amaç: ∑ xi^2 fonksiyonunu minimize etmek.
    """
    karar_degerleri = birey_to_real(birey)
    fitness = -np.sum(karar_degerleri ** 2)  # Minimizasyon için negatif işaret
    return fitness

def birey_to_real(birey):
    """
    Bireyi 75 bitlik ikili temsilinden [-100, 100] reel sayılarına çevirir.
    """
    reel_sayilar = []
    bit_boyutu = birey_uzunlugu // problem_boyutu
    for i in range(problem_boyutu):
        alt_bolge = birey[i * bit_boyutu: (i + 1) * bit_boyutu]
        reel_deger = int("".join(map(str, alt_bolge)), 2)
        reel_sayilar.append(-100 + (reel_deger / (2 ** bit_boyutu - 1)) * 200)
    return np.array(reel_sayilar)

def rulet_tekerlegi_secim(populasyon, fitness_degerleri):
    toplam_fitness = np.sum(fitness_degerleri)
    secim_olasiliklari = fitness_degerleri / toplam_fitness
    kumulatif_olasiliklar = np.cumsum(secim_olasiliklari)

    yeni_populasyon = []
    for _ in range(populasyon_buyuklugu):
        rastgele_sayi = np.random.rand()
        secilen_birey = np.where(kumulatif_olasiliklar >= rastgele_sayi)[0][0]
        yeni_populasyon.append(populasyon[secilen_birey])

    return np.array(yeni_populasyon)

def caprazlama(ebeveyn1, ebeveyn2):
    """
    İki ebeveyn bireyden tek noktalı çaprazlama ile yeni birey oluşturur.
    """
    if np.random.rand() < caprazlama_olasiligi:
        caprazlama_noktasi = np.random.randint(1, birey_uzunlugu)
        cocuk1 = np.concatenate([ebeveyn1[:caprazlama_noktasi], ebeveyn2[caprazlama_noktasi:]])
        cocuk2 = np.concatenate([ebeveyn2[:caprazlama_noktasi], ebeveyn1[caprazlama_noktasi:]])
        return cocuk1, cocuk2
    return ebeveyn1, ebeveyn2

def mutasyon(birey):
    """
    Mutasyon olasılığına bağlı olarak bireyin genlerini değiştirir.
    """
    for i in range(birey_uzunlugu):
        if np.random.rand() < mutasyon_olasiligi:
            birey[i] = 1 - birey[i]  # 0 -> 1 veya 1 -> 0 dönüşümü
    return birey

for iterasyon in range(iterasyon_sayisi):

    fitness_degerleri = np.array([fitness_hesapla(birey) for birey in populasyon])

    populasyon = rulet_tekerlegi_secim(populasyon, fitness_degerleri)

    yeni_populasyon = []
    for i in range(0, populasyon_buyuklugu, 2):
        ebeveyn1, ebeveyn2 = populasyon[i], populasyon[i + 1]
        cocuk1, cocuk2 = caprazlama(ebeveyn1, ebeveyn2)
        yeni_populasyon.extend([mutasyon(cocuk1), mutasyon(cocuk2)])

    populasyon = np.array(yeni_populasyon)

    en_iyi_fitness = np.max(fitness_degerleri)
    en_iyi_birey = populasyon[np.argmax(fitness_degerleri)]
    en_iyi_karar_degerleri = birey_to_real(en_iyi_birey)
    print(f"Iterasyon {iterasyon + 1}, En İyi Fitness: {en_iyi_fitness}, Karar Değerleri: {en_iyi_karar_degerleri}")

print("Genetik algoritma tamamlandı.")


