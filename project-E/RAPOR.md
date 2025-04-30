# Model Predictive Control (MPC) - Öğrenim ve Uygulama Raporu

---

## 1. Giris

Model Predictive Control (MPC), sistemin gelecekteki davranışını tahmin ederek en uygun kontrol sinyalini belirleyen ileri seviye bir kontrol yöntemidir.
MPC, sistem kısıtlarını (örn: maksimum voltaj, maksimum hız) doğrudan hesaba katabilir ve özellikle elektrikli motorlar, otonom sistemler ve enerji yönetimi alanlarında yaygın olarak kullanılır.

---

## 2. DC Motor Sistem Modeli

### Durum Uzayı Temsili:

- Elektriksel Dinamik:
  \[ V(t) = L \frac{di(t)}{dt} + Ri(t) + K_e \omega(t) \]
- Mekanik Dinamik:
  \[ J \frac{d\omega(t)}{dt} + B\omega(t) = K_t i(t) \]

### Durum Değişkenleri:
- \( x_1 = i(t) \) : Akım
- \( x_2 = \omega(t) \) : Açısal Hız

### Durum Uzayı Modeli:

\[
\dot{x} = A x + B u
\]
\[
y = C x + D u
\]

Matrisler:

\[
A = \begin{bmatrix} -\frac{R}{L} & -\frac{K_e}{L} \\ \frac{K_t}{J} & -\frac{B}{J} \end{bmatrix}, \quad B = \begin{bmatrix} \frac{1}{L} \\ 0 \end{bmatrix}, \quad C = \begin{bmatrix} 0 & 1 \end{bmatrix}, \quad D = 0
\]

Sistem MATLAB ortamında diskretize edilerek MPC tasarımı için hazırlandı.

---

## 3. Temel 1-Adım Horizon MPC

- **Prediction Horizon:** 1 adım.
- **Cost Fonksiyonu:**
  \[ J = (y_{k+1} - r)^2 + \lambda u_k^2 \]

### Özellikler:
- Her adımda sadece bir adım ötesi tahmin edildi.
- Optimal kontrol sinyali analitik olarak çözüldü.
- Kontrol ceza katsayısı (lambda) ayarlanarak sistem davranışı optimize edildi.

### Sonuçlar:
- Sistem stabil bir şekilde hedef hıza ulaştı.
- Kontrol kuvvetinin etkisi doğrudan gözlemlendi.

---

## 4. Multi-Step (5 Adım Horizon) MPC

- **Prediction Horizon:** 5 adım.
- **Optimization Yöntemi:** Quadratic Programming (QP) kullanıldı (`quadprog`).
- **Cost Fonksiyonu:**
  \[ J = \sum_{i=1}^{5} \left( (y_{k+i} - r)^2 + \lambda (u_{k+i-1})^2 \right) \]

### Özellikler:
- Gelecek 5 adım boyunca sistemin davranışı tahmin edildi.
- Optimal voltaj dizisi hesaplandı.
- Sadece ilk kontrol sinyali uygulandı (receding horizon mantığı).

### Sonuçlar:
- Sistem hedefe daha hızlı ve optimize edilmiş bir yol izleyerek ulaştı.
- Voltaj kısıtları ([-5V, +5V]) başarıyla uygulandı.
- Overshoot ve dalgalanma gözlenmedi.

---

## 5. Sonuçlar ve Gözlemler

- 1 adım MPC basit ve doğrudan, ancak çok akıllı kararlar veremeyen bir yapı sundu.
- Multi-Step MPC, daha planlı ve geleceği optimize ederek sistem davranışını iyileştirdi.
- Kontrol ceza katsayısı (lambda) sistemin agresiflik veya sakinliği üzerinde doğrudan etkili oldu.

---

## 6. Kazanımlar ve Gelecek Adımlar

### Kazanımlar:
- MPC'nin temel mekanizması ve tahmin modeli anlaşıldı.
- Diskretize sistem kullanarak MPC kurulumu öğrenildi.
- Quadratic Programming ile optimal kontrol hesaplandı.
- Sistem kısıtlarını MPC içine entegre etme pratiği kazanıldı.

### Potansiyel Gelişimler:
- Prediction horizon arttırılarak daha akıllı kontrol.
- Soft constraints eklenerek daha esnek sistem yönetimi.
- Multi-Input Multi-Output (MIMO) sistemlerde MPC uygulaması.
- Realtime MPC simülasyonları ve donanım üzerinde uygulamalar.
- Neural Network destekli öğrenen MPC geliştirilmesi.

---

## 7. Neural Network Destekli MPC

### Veri Toplama:
- DC motor sistemine rastgele girişler uygulanarak giriş-çıkış verileri toplandı.
- Girişler: Voltaj (u)
- Çıkışlar: Açısal hız (ω)

### Neural Network Modeli:
- Giriş: [Akım (i), Açısal Hız (ω), Giriş Voltajı (u)]
- Çıkış: Bir sonraki adım Açısal Hız (ω_{t+1})
- Yapı: PyTorch kullanılarak 2 gizli katmanlı küçük bir MLP ağı kuruldu.

### Eğitim:
- MSE Loss kullanılarak model eğitildi.
- Eğitim sonucunda model sistem dinamiklerini başarılı şekilde öğrendi.

### Neural Network ile MPC:
- Klasik MPC'nin fiziksel model yerine Neural Network tahmin modeli kullanıldı.
- Tahminler her adımda NN üzerinden yapılarak optimizasyon gerçekleştirildi.

### Gözlemler:
- Neural Network destekli MPC, fiziksel modele yakın performans gösterdi.
- Karmaşık sistemlerde modelleme zorluklarını aşmak için etkili bir yöntem olduğu gözlemlendi.
