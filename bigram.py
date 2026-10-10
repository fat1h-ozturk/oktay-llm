"""Önceki karakterden sıradaki karakteri öğrenen küçük CPU modeli."""

from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from tokenizer import KarakterTokenizer


class BigramModel(nn.Module):
    def __init__(self, sozluk_boyutu: int):
        super().__init__()
        # Her satır bir önceki karakter için sonraki karakterlerin puanlarını tutar.
        self.tablo = nn.Embedding(sozluk_boyutu, sozluk_boyutu)

    def forward(self, numaralar: torch.Tensor) -> torch.Tensor:
        return self.tablo(numaralar)

    @torch.no_grad()
    def generate(self, baslangic: torch.Tensor, yeni_karakter_sayisi: int) -> torch.Tensor:
        if baslangic.ndim != 1 or baslangic.numel() == 0:
            raise ValueError("Başlangıç, en az bir karakter içeren tek boyutlu tensor olmalı.")
        if yeni_karakter_sayisi < 0:
            raise ValueError("Yeni karakter sayısı negatif olamaz.")
        numaralar = baslangic.clone()
        for _ in range(yeni_karakter_sayisi):
            puanlar = self(numaralar[-1:])
            olasiliklar = F.softmax(puanlar, dim=-1)
            sonraki = torch.multinomial(olasiliklar, num_samples=1).reshape(1)
            numaralar = torch.cat((numaralar, sonraki))
        return numaralar


if __name__ == "__main__":
    torch.manual_seed(42)
    # Bu çok küçük hesaplamada thread yönetimi maliyeti işi yavaşlatabilir.
    torch.set_num_threads(1)
    metin = (Path(__file__).parent / "data" / "ornek.txt").read_text(encoding="utf-8")
    if len(metin) < 2:
        raise ValueError("Eğitim için en az iki karakter gerekir.")
    tokenizer = KarakterTokenizer(metin)
    veri = torch.tensor(tokenizer.encode(metin), dtype=torch.long)

    # Örnek: 'Bir' için girdiler 'Bi', hedefler 'ir' olur.
    girdiler, hedefler = veri[:-1], veri[1:]
    model = BigramModel(len(tokenizer.karakterler))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    print("Cihaz: CPU")
    print("Parametre sayısı:", sum(p.numel() for p in model.parameters()))
    print("Karakter çifti sayısı:", len(girdiler))

    baslama = perf_counter()
    model.train()
    for adim in range(301):
        puanlar = model(girdiler)
        hata = F.cross_entropy(puanlar, hedefler)
        if adim % 100 == 0:
            print(f"Adım {adim:3d} | Eğitim hatası: {hata.item():.4f}")
        if adim < 300:
            optimizer.zero_grad()
            hata.backward()
            optimizer.step()
    print(f"Eğitim süresi: {perf_counter() - baslama:.2f} saniye")

    model.eval()
    baslangic = torch.tensor(tokenizer.encode(metin[:3]), dtype=torch.long)
    baslama = perf_counter()
    uretilen = model.generate(baslangic, 200)
    print(f"200 karakter üretim süresi: {perf_counter() - baslama:.2f} saniye")
    print("\nÜretilen metin:")
    print(tokenizer.decode(uretilen.tolist()))
    print("\nBu sonuç eğitim metnindeki hatadır; bağımsız test başarısı ölçülmedi.")
