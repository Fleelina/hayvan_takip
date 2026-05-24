"""
Test videosu oluşturucu.
Gerçek video olmadan sistemi test etmek için sahte hareketli hayvan kutuları çizer.
Kullanım: python create_test_video.py
"""

import cv2
import numpy as np
import os
import math
import random

# ── Ayarlar ──────────────────────────────────────────────────────────────────
OUTPUT_PATH = "videos/input_video.mp4"
WIDTH, HEIGHT = 1280, 720
FPS = 25
DURATION_SEC = 30          # 30 saniyelik video
TOTAL_FRAMES = FPS * DURATION_SEC

# COCO'da var olan çiftlik hayvanı renkleri (BGR)
ANIMAL_COLORS = {
    "Cow":   (0, 200, 255),
    "Sheep": (0, 255, 100),
    "Horse": (255, 150, 0),
}

random.seed(42)

# ── Hayvan tanımları ──────────────────────────────────────────────────────────
# Her hayvan: başlangıç x,y, hız vx/vy, boyut, tür
def make_animals():
    animals = []
    specs = [
        # (x0, y0, vx, vy, w, h, kind)
        (100,  200,  3.5,  0.8, 120, 80, "Cow"),
        (900,  400, -2.8,  1.2, 110, 75, "Sheep"),
        (400,  100,  1.2,  4.0,  90, 65, "Sheep"),
        (1100, 300, -3.0, -0.5, 130, 85, "Horse"),
        (200,  600,  4.0, -1.5, 115, 78, "Cow"),
        (600,  500, -1.5,  2.5,  95, 68, "Sheep"),
        (50,   350,  2.0,  3.0, 108, 72, "Cow"),
        (1000, 150, -2.5,  2.0,  88, 60, "Sheep"),
    ]
    for x, y, vx, vy, w, h, kind in specs:
        animals.append({
            "x": float(x), "y": float(y),
            "vx": vx, "vy": vy,
            "w": w, "h": h,
            "kind": kind,
            "color": ANIMAL_COLORS[kind],
            "wobble": random.uniform(0, math.pi * 2),   # hareket sallantısı
        })
    return animals

# ── Arka plan çizimi ─────────────────────────────────────────────────────────
def draw_background(frame):
    # Yeşil çimen zemin
    frame[:] = (34, 85, 34)
    # Basit gökyüzü şeridi (üstte)
    frame[:80, :] = (160, 120, 80)
    # Ahır silueti (sağda)
    pts = np.array([[1000, 80], [1050, 30], [1100, 80], [1100, 200], [1000, 200]], np.int32)
    cv2.fillPoly(frame, [pts], (50, 60, 120))
    # Zemin çizgisi
    cv2.line(frame, (0, 80), (WIDTH, 80), (90, 110, 60), 2)
    # Çim dokusu (rastgele kısa çizgiler)
    rng = np.random.default_rng(0)
    for _ in range(300):
        gx = int(rng.integers(0, WIDTH))
        gy = int(rng.integers(82, HEIGHT))
        cv2.line(frame, (gx, gy), (gx + rng.integers(-4, 4), gy - rng.integers(4, 10)),
                 (40, 100, 40), 1)

# ── Hayvan kutusu çizimi ─────────────────────────────────────────────────────
def draw_animal(frame, a, frame_idx):
    x1 = int(a["x"] - a["w"] // 2)
    y1 = int(a["y"] - a["h"] // 2)
    x2 = x1 + a["w"]
    y2 = y1 + a["h"]
    color = a["color"]

    # Gövde (elips)
    cx, cy = int(a["x"]), int(a["y"])
    wobble = int(3 * math.sin(frame_idx * 0.15 + a["wobble"]))
    cv2.ellipse(frame, (cx, cy + wobble), (a["w"] // 2, a["h"] // 2),
                0, 0, 360, color, -1)
    # Baş
    head_r = a["h"] // 4
    head_x = cx + (a["w"] // 2 - head_r) * (1 if a["vx"] >= 0 else -1)
    cv2.circle(frame, (head_x, cy - a["h"] // 4), head_r, color, -1)

    # Bacaklar (4 adet basit çizgi)
    leg_y = cy + a["h"] // 2 + wobble
    step = int(4 * math.sin(frame_idx * 0.25 + a["wobble"]))
    for lx_offset in [-a["w"] // 3, -a["w"] // 8, a["w"] // 8, a["w"] // 3]:
        cv2.line(frame, (cx + lx_offset, cy + a["h"] // 2),
                 (cx + lx_offset, leg_y + step * (1 if lx_offset < 0 else -1) + 20),
                 color, 3)

    # Hafif gölge
    shadow_pts = np.array([
        [x1 + 10, y2], [x2 - 10, y2], [x2, y2 + 8], [x1, y2 + 8]
    ], np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [shadow_pts], (20, 50, 20))
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)


# ── Ana döngü ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs("videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    animals = make_animals()
    base_bg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    draw_background(base_bg)

    print(f"[BİLGİ] Test videosu oluşturuluyor: {OUTPUT_PATH}")
    print(f"        {WIDTH}x{HEIGHT} @ {FPS} FPS, {DURATION_SEC} saniye ({TOTAL_FRAMES} kare)")

    for f_idx in range(TOTAL_FRAMES):
        frame = base_bg.copy()

        for a in animals:
            # Hareket + sınır sekme
            a["x"] += a["vx"]
            a["y"] += a["vy"] + math.sin(f_idx * 0.05 + a["wobble"]) * 0.4

            if a["x"] - a["w"] // 2 < 0 or a["x"] + a["w"] // 2 > WIDTH:
                a["vx"] *= -1
            if a["y"] - a["h"] // 2 < 80 or a["y"] + a["h"] // 2 > HEIGHT - 10:
                a["vy"] *= -1

            draw_animal(frame, a, f_idx)

        # İlerleme
        if f_idx % (FPS * 5) == 0:
            sec = f_idx // FPS
            print(f"  {sec}/{DURATION_SEC} saniye işlendi...")

        out.write(frame)

    out.release()
    print(f"\n✅ Test videosu hazır: {OUTPUT_PATH}")
    print("   Şimdi çalıştırabilirsiniz:")
    print("   python main.py --input videos/input_video.mp4")


if __name__ == "__main__":
    main()
