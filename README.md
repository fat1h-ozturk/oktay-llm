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

## İkinci adım: karakter tokenizer

```bash
.venv/bin/python tokenizer.py
.venv/bin/python -m unittest discover -v
```

Windows'ta aynı komutlarda `.venv/bin/python` yerine
`.\.venv\Scripts\python.exe` kullan.

`data/ornek.txt`, bu proje için yazılmış kısa bir deneme metnidir; model
eğitmek için yeterli bir veri kümesi değildir. UTF-8 olarak okunur; boşluklar,
noktalama işaretleri ve satır sonları korunur.

`tokenizer.py` içindeki `KarakterTokenizer` metindeki farklı karakterleri
sıralayıp her birine 0'dan başlayan bir numara verir. Örneğin `baba` metninde
`a → 0`, `b → 1` olur ve `encode("baba")` sonucu `[1, 0, 1, 0]` çıkar.
`decode([1, 0, 1, 0])` tekrar `baba` verir. Bunlar token numaralarıdır;
her numara tek bir karakteri temsil eder.

Sözlükte olmayan bir karakter ya da geçersiz bir numara açık hata verir;
metin sessizce değiştirilmez. Yeni metinlerle aynı modeli kullanırken aynı
karakter sözlüğünü kullanmak gerekir. Bu örnekte sözlük her çalıştırmada
örnek metinden oluşturulur; kalıcı sözlük kaydı eğitim aşamasında eklenecek.

Alıştırma: `data/ornek.txt` dosyasına bir Türkçe cümle ekle ve kodu yeniden
çalıştır. Metnin uzunluğu ve farklı karakter sayısının nasıl değiştiğini incele.

## Üçüncü adım: ilk dil modeli (bigram)

```bash
.venv/bin/python bigram.py
.venv/bin/python -m unittest discover -v
```

`bigram.py`, örnek metindeki ardışık karakter çiftleriyle CPU'da 300 eğitim
adımı çalıştırır. Eğitim hatasını, süreyi, parametre sayısını ve 200 yeni
karakterden oluşan bir devam metnini gösterir. Her çalıştırmada model yeniden
eğitilir; bu aşamada model dosyası kaydedilmez.

Model yalnızca önceki karakteri kullanır. Örneğin `Bir` kelimesinden `B → i`
ve `i → r` eğitim çiftleri çıkar. Her karakter için olası sonraki karakterlerin
puanları öğrenilir; softmax bu puanları olasılıklara çevirir.

Eğitim döngüsünün dört temel adımı:

1. `model(girdiler)`: sıradaki karakterlerin puanlarını hesaplar.
2. `cross_entropy`: puanlarla doğru karakterleri karşılaştırıp hatayı hesaplar.
3. `hata.backward()`: ağırlıkların hatayı nasıl etkilediğini hesaplar.
4. `optimizer.step()`: ağırlıkları günceller. `zero_grad()` eski türevleri temizler.

Üretimde ağırlıklar değişmez. Model son karakterden tahmin yapar, olasılıklara
göre yeni bir karakter seçer ve bunu metnin sonuna ekler. Başlangıç metni örnek
dosyanın ilk üç karakteridir. Rastgelelik tohumu, CPU denemelerini aynı ortamda
tekrarlamayı kolaylaştırmak için 42 olarak ayarlanmıştır.

Örnek metinle model 1.936 parametre içerir. Bu, ileride yazacağımız yaklaşık
1 milyon parametreli Transformer'dan çok daha küçüktür. Bigram çıktısının
anlamsız olması beklenir: model önceki cümleyi ya da kelimeyi hatırlamaz.

Bu denemede verinin tamamı eğitim için kullanılır. Gösterilen hata **eğitim
hatasıdır**; modelin yeni metinlerde başarısını göstermez. Gerçek veri kümesini
hazırlarken belgeleri eğitim, doğrulama ve test gruplarına ayıracağız.

Alıştırma: öğrenme oranını (`lr=0.1`) değiştir ve eğitim hatasının nasıl
değiştiğini incele. Aynı anda başka bir ayarı değiştirme.

## Teslim için model boyutu hedefi

Transformer için başlangıç hedefi yaklaşık 1 milyon parametre: 4 katman,
128 temsil boyutu, 4 dikkat başlığı ve 128 karakter bağlam. Üst sınır 5 milyon
parametre; büyütme kararı CPU ölçümlerine bağlı olacak. Demo için 200 karakteri
10 saniye içinde üretmek bir hedeftir; Transformer yazıldıktan sonra ölçülecek.
Eğitim RTX 5070'li bilgisayarda, demo CPU'da çalışabilecek şekilde hazırlanacak.
Hocanın işletim sistemi ve donanımı henüz bilinmiyor; teslim öncesi o bilgisayarda
kurulum ve hız doğrulanmalı. Bigram süreleri Transformer hızını göstermez.

Sıradaki aşama, daha geniş bağlamı kullanabilmek için dikkat mekanizmasını
öğrenmek ve küçük Transformer'a geçmek.

Kurulum kaynağı: [PyTorch resmi kurulum sayfası](https://pytorch.org/get-started/locally/).
