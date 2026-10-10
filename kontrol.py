"""PyTorch kurulumu, CPU hesaplaması ve otomatik türev kontrolü."""

import torch


print("PyTorch sürümü:", torch.__version__)
print("CUDA kullanılabilir:", torch.cuda.is_available())

# Tensor, modelin hesaplamalarında kullandığımız sayı dizisidir.
sayilar = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
sonuc = sayilar * 2
torch.testing.assert_close(sonuc, torch.tensor([2.0, 4.0, 6.0]))
print("İlk hesaplama:", sonuc.detach())

# x²'nin türevi 2x'tir; backward bu türevi otomatik hesaplar.
hata = (sayilar ** 2).sum()
hata.backward()
torch.testing.assert_close(sayilar.grad, torch.tensor([2.0, 4.0, 6.0]))
print("Hesaplanan türev:", sayilar.grad)
print("CPU hesaplaması ve türev kontrolü başarılı!")
