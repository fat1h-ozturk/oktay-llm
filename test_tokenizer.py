import unittest

from tokenizer import KarakterTokenizer


class KarakterTokenizerTest(unittest.TestCase):
    def test_karakterler_tekrar_kullanimda_ayni_numarayi_alir(self):
        tokenizer = KarakterTokenizer("baba")
        self.assertEqual(tokenizer.encode("baba"), [1, 0, 1, 0])
        self.assertEqual(tokenizer.decode([1, 0, 1, 0]), "baba")

    def test_turkce_harfler_bosluk_ve_satir_sonu_korunur(self):
        metin = "Çığ, şÖğÜİı!\nİkinci satır.  "
        tokenizer = KarakterTokenizer(metin)
        self.assertEqual(tokenizer.decode(tokenizer.encode(metin)), metin)
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.decode([]), "")

    def test_bilinmeyen_karakter_sessizce_atilmaz(self):
        tokenizer = KarakterTokenizer("abc")
        with self.assertRaises(ValueError):
            tokenizer.encode("aç")

    def test_gecersiz_numara_baska_karaktere_donusmez(self):
        tokenizer = KarakterTokenizer("abc")
        for numara in [-1, 3]:
            with self.subTest(numara=numara), self.assertRaises(ValueError):
                tokenizer.decode([numara])

    def test_bos_metinden_sozluk_olusturulamaz(self):
        with self.assertRaises(ValueError):
            KarakterTokenizer("")


if __name__ == "__main__":
    unittest.main()
