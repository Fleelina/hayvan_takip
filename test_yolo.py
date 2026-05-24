"""
YOLO Bağlantı Testi — İnternetten örnek hayvan görseli indirip test eder.
Kullanım: python test_yolo.py
"""

import urllib.request
import cv2
import os
from ultralytics import YOLO

# Test için kullanılacak örnek görsel URL'leri (açık lisanslı)
TEST_IMAGES = [
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Cow_female_black_white.jpg/800px-Cow_female_black_white.jpg",
        "test_cow.jpg"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Domestic_goat_kid_in_capeweed.jpg/800px-Domestic_goat_kid_in_capeweed.jpg",
        "test_sheep.jpg"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/800px-Collage_of_Nine_Dogs.jpg",
        "test_dog.jpg"
    ),
]

FARM_ANIMALS = {"cow", "sheep", "horse", "dog", "cat", "bird", "goat"}

CLASS_COLORS = {
    "cow":   (0, 200, 255),
    "sheep": (0, 255, 100),
    "horse": (255, 150, 0),
    "dog":   (200, 0, 255),
    "cat":   (255, 0, 150),
    "bird":  (0, 255, 255),
}

def download_image(url, path):
    if os.path.exists(path):
        return True
    try:
        print(f"  İndiriliyor: {url[:60]}...")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            with open(path, "wb") as f:
                f.write(r.read())
        return True
    except Exception as e:
        print(f"  [UYARI] İndirilemedi: {e}")
        return False

def test_image(model, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [HATA] Görüntü okunamadı: {img_path}")
        return

    results = model(img, conf=0.25, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_name = r.names[int(box.cls)].lower()
            conf_val = float(box.conf[0])
            detections.append((cls_name, conf_val))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf_val:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    os.makedirs("outputs", exist_ok=True)
    out_name = "outputs/test_" + os.path.basename(img_path)
    cv2.imwrite(out_name, img)

    if detections:
        print(f"  ✅ Tespit edildi → {[(d[0], f'{d[1]:.2f}') for d in detections]}")
        print(f"     Kaydedildi   → {out_name}")
    else:
        print(f"  ⚠️  Hiçbir şey tespit edilemedi. (Görselde hayvan var mı?)")
        print(f"     Kaydedildi   → {out_name}")

def main():
    print("=" * 55)
    print("  YOLO HAYVAN TESPİT TESTİ")
    print("=" * 55)
    print("\n[1] Model yükleniyor: yolov8n.pt")
    model = YOLO("yolov8n.pt")
    print("    Model hazır.\n")

    print("[2] Test görselleri indiriliyor ve test ediliyor...\n")
    for url, filename in TEST_IMAGES:
        print(f"── {filename}")
        if download_image(url, filename):
            test_image(model, filename)
        print()

    print("[3] Sonuç görselleri 'outputs/' klasörüne kaydedildi.")
    print("\nEğer tespitler başarılıysa sistem çalışıyor demektir.")
    print("Gerçek bir çiftlik videosu ile deneyin:")
    print("  python main.py --input videos/input_video.mp4")
    print("=" * 55)

if __name__ == "__main__":
    main()
