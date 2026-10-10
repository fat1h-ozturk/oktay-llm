# Sıfırdan Türkçe Mini GPT

İleri programlama dersi için Türkçe metin tamamlayan küçük bir dil modeli
geliştireceğiz. Model hazır ağırlık kullanmadan eğitilecek. Haftalık kapsam
`Turkce_Mini_GPT_Proje_Plani.docx` belgesinde bulunuyor.

## İlk adım: CPU ortamı

Bu bilgisayarda Linux ve Python 3.14 kullanıyoruz. Proje klasöründe:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python kontrol.py
```

Ortam zaten oluşturulduysa ilk iki komutu tekrar çalıştırman gerekmez.
VS Code kullanıyorsan Python yorumlayıcısı olarak `.venv/bin/python` seç.

Windows'ta CPU denemesi için:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe kontrol.py
```

`requirements-cpu.txt` işlemci sürümünü kurar. RTX 5070 ile eğitim yapacağımız
bilgisayarın GPU kurulumu ayrıca hazırlanacak.

## Kontrol kodu ne yapıyor?

`kontrol.py`, `[1, 2, 3]` sayılarını bir PyTorch tensorüne dönüştürür.
Tensorü 2 ile çarparak `[2, 4, 6]` sonucunu kontrol eder.
Ardından karelerin toplamının türevini `backward()` ile hesaplar ve yine
`[2, 4, 6]` olduğunu doğrular. Eğitimde bu mekanizma, hatanın modelin
ağırlıklarına göre nasıl değiştiğini hesaplamamızı sağlar.

Son satırda `CPU hesaplaması ve türev kontrolü başarılı!` görmelisin.
Bu CPU ortamında `CUDA kullanılabilir: False` beklenen sonuçtur.

İlk alıştırma: kodu incele, sayıların değerlerini değiştir ve beklenen sonuçları
kontrol eden satırları buna göre güncelle.

## Sonraki adım

UTF-8 Türkçe metin okuma ve karakterleri sayılara, sayıları tekrar metne
dönüştürme üzerinde çalışacağız. Model geliştirmeden önce bu dönüşümü anlamak
önemli.

Kurulum kaynağı: [PyTorch resmi kurulum sayfası](https://pytorch.org/get-started/locally/).
