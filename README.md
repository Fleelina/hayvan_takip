# 🐄 Hayvan Takip ve Giriş/Çıkış Sayma Sistemi

YOLOv8 + ByteTrack kullanarak çiftlik hayvanlarını tespit eden, takip eden ve ekrandan giriş/çıkışlarını sayan sistem.

---

## 📁 Klasör Yapısı

```
hayvan_takip/
├── main.py              ← Ana program
├── requirements.txt     ← Gerekli paketler
├── videos/
│   └── input_video.mp4  ← Buraya videonuzu ekleyin
├── outputs/
│   └── result.mp4       ← İşlenmiş video burada çıkar
└── README.md
```

---

## ⚙️ Kurulum

```bash
# 1. Gerekli paketleri yükle
pip install -r requirements.txt

# 2. Videonuzu videos/ klasörüne koyun:
#    videos/input_video.mp4
```

> İlk çalıştırmada `yolov8s.pt` modeli (~22 MB) otomatik indirilir. Daha hafif için `--model yolov8n.pt`, daha yüksek doğruluk için `--model yolov8m.pt` kullanabilirsiniz.

---

## ▶️ Çalıştırma

### Temel kullanım
```bash
python main.py --input videos/input_video.mp4
```

### Tüm seçenekler
```bash
python main.py \
  --input videos/input_video.mp4 \
  --output outputs/result.mp4 \
  --model yolov8n.pt \
  --conf 0.45 \
  --classes cow sheep horse \
  --skip-frames 1
```

---

## 🎛️ Parametre Açıklamaları

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `--input` | `videos/input_video.mp4` | Giriş video yolu |
| `--output` | `outputs/result.mp4` | Çıkış video yolu |
| `--model` | `yolov8s.pt` | YOLO model dosyası (`yolov8n`, `yolov8s`, `yolov8m`) |
| `--conf` | `0.45` | Minimum güven eşiği |
| `--classes` | tüm çiftlik hayvanları | Sadece bu türleri say |
| `--skip-frames` | `1` | Her N karede bir işle (zayıf CPU için 2–3 deneyin) |
| `--no-show` | kapalı | Ekranda gösterme, sadece kaydet |

---

## 🐑 Hayvan Filtresi

Yalnızca belirli hayvanları saymak için `--classes` kullanın:

```bash
# Sadece inek ve koyun
python main.py --input videos/input_video.mp4 --classes cow sheep

# Sadece at
python main.py --input videos/input_video.mp4 --classes horse

# Tüm çiftlik hayvanları (varsayılan)
python main.py --input videos/input_video.mp4
```

**Desteklenen sınıflar:** `cow` `sheep` `horse` `dog` `cat` `bird`

---

## 🖥️ Ekran Göstergeleri

- **Sol üst panel:** FPS, aktif tespit sayısı ve toplam tekil ID
- **Renkli bounding box + etiket:** Tür, ID, güven oranı
- **Hareket izi:** Hayvanın son 40 karesinin soluklaşan renk izi
- **Alt progress bar:** Videonun işlenme ilerlemesi

---

| Model | Hız | Doğruluk | Boyut |
|-------|------|----------|-------|
| `yolov8n.pt` | çok hızlı | düşük | ~6 MB |
| `yolov8s.pt` | hızlı | iyi (**varsayılan**) | ~22 MB |
| `yolov8m.pt` | orta | yüksek | ~52 MB |

---

## 💻 Performans İpuçları

Zayıf bilgisayarlarda:
```bash
# Daha küçük model (daha hızlı, biraz daha az hassas)
python main.py --model yolov8n.pt --skip-frames 2

# Ekranda göstermeden sadece kaydet (daha hızlı)
python main.py --no-show --skip-frames 2
```

---

## 📦 Kullanılan Teknolojiler

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack) (Ultralytics içinde dahili)
- OpenCV
- NumPy
