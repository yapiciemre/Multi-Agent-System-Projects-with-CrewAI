# Python 3.11.5 slim imajını temel alarak Docker imajını oluşturuyoruz.
FROM python:3.11.5-slim

# Çalışma dizinini /app olarak belirliyoruz. Bu dizin, Docker konteyneri içinde çalışacak tüm dosyalar için temel dizin olacaktır.
WORKDIR /app

# requirements.txt dosyasını çalışma dizinine kopyalıyoruz. Bu dosya, proje bağımlılıklarını listeler.
COPY requirements.txt .

# requirements.txt içindeki bağımlılıkları yüklüyoruz.
# --no-cache-dir bayrağı, yükleme sırasında herhangi bir önbellek kullanılmamasını sağlar, bu da imajın daha az yer kaplamasını sağlar.
RUN pip install --no-cache-dir -r requirements.txt

# Projedeki tüm dosyaları çalışma dizinine kopyalıyoruz. Bu adım, uygulama kodunun Docker imajına dahil edilmesini sağlar.
COPY . .

# Uygulamayı başlatmak için kullanılacak komutu belirliyoruz.
# CMD ["python", "main.py"]
