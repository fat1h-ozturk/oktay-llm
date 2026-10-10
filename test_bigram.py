import unittest

import torch
from torch.nn import functional as F

from bigram import BigramModel


class BigramModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.onceki_thread_sayisi = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.onceki_thread_sayisi)

    def test_egitim_siradaki_karakteri_ogrenir(self):
        torch.manual_seed(42)
        model = BigramModel(2)
        girdiler = torch.tensor([0, 1, 0, 1])
        hedefler = torch.tensor([1, 0, 1, 0])
        ilk_hata = F.cross_entropy(model(girdiler), hedefler).item()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        for _ in range(40):
            hata = F.cross_entropy(model(girdiler), hedefler)
            optimizer.zero_grad()
            hata.backward()
            optimizer.step()
        son_hata = F.cross_entropy(model(girdiler), hedefler).item()
        self.assertLess(son_hata, ilk_hata / 4)
        self.assertEqual(model(girdiler).argmax(dim=-1).tolist(), [1, 0, 1, 0])

    def test_uretim_baslangici_korur_ve_son_karakteri_kullanir(self):
        model = BigramModel(2)
        with torch.no_grad():
            model.tablo.weight.copy_(torch.tensor([[-100.0, 100.0], [100.0, -100.0]]))
        baslangic = torch.tensor([1, 1, 0])
        sonuc = model.generate(baslangic, 4)
        self.assertEqual(sonuc.tolist(), [1, 1, 0, 1, 0, 1, 0])
        self.assertEqual(baslangic.tolist(), [1, 1, 0])

    def test_uretim_icin_bos_baslangic_reddedilir(self):
        with self.assertRaises(ValueError):
            BigramModel(2).generate(torch.tensor([], dtype=torch.long), 4)

    def test_negatif_uretim_uzunlugu_reddedilir(self):
        with self.assertRaises(ValueError):
            BigramModel(2).generate(torch.tensor([0]), -1)


if __name__ == "__main__":
    unittest.main()
