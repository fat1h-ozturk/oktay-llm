"""Türkçe metni karakter numaralarına çevirme ve geri okuma."""

from pathlib import Path


class KarakterTokenizer:
    def __init__(self, metin: str):
        if not metin:
            raise ValueError("Karakter sözlüğü için metin boş olamaz.")
        self.karakterler = sorted(set(metin))
        self.karakterden_sayiya = {
            karakter: numara for numara, karakter in enumerate(self.karakterler)
        }
        self.sayidan_karaktere = {
            numara: karakter for numara, karakter in enumerate(self.karakterler)
        }

    def encode(self, metin: str) -> list[int]:
        try:
            return [self.karakterden_sayiya[karakter] for karakter in metin]
        except KeyError as hata:
            raise ValueError(f"Sözlükte bulunmayan karakter: {hata.args[0]!r}") from hata

    def decode(self, numaralar: list[int]) -> str:
        try:
            return "".join(self.sayidan_karaktere[numara] for numara in numaralar)
        except KeyError as hata:
            raise ValueError(f"Sözlükte bulunmayan numara: {hata.args[0]!r}") from hata


if __name__ == "__main__":
    dosya = Path(__file__).parent / "data" / "ornek.txt"
    metin = dosya.read_text(encoding="utf-8")
    tokenizer = KarakterTokenizer(metin)
    numaralar = tokenizer.encode(metin)
    geri_okunan = tokenizer.decode(numaralar)
    assert geri_okunan == metin, "Geri okunan metin başlangıç metninden farklı!"

    print("Metindeki karakter sayısı:", len(metin))
    print("Farklı karakter sayısı:", len(tokenizer.karakterler))
    print("Karakter sözlüğü:", tokenizer.karakterden_sayiya)
    print("İlk 40 karakter:", repr(metin[:40]))
    print("İlk 40 karakterin numaraları:", numaralar[:40])
    print("Geri okunan ilk 40 karakter:", repr(geri_okunan[:40]))
    print("Metin → sayılar → metin kontrolü başarılı!")
