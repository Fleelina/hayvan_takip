"""
============================================================
Hayvan Tespit ve Sayım Sistemi
============================================================
Kullanım:
    python main.py --input videos/input_video.mp4
"""

import cv2
import numpy as np
import argparse
import time
import os
import torch
import clip
from PIL import Image
from collections import defaultdict
from ultralytics import YOLO

CONF_HIGH       = 0.75
CONF_LOW        = 0.40
CLIP_THRESH     = 0.35  # CLIP inek skoru eşiği
VERIFY_CACHE_FRAMES = 300

# CLIP modelini yükle (ilk seferde ~150MB indirir)
print("[BİLGİ] CLIP yükleniyor...")
_clip_model, _clip_preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
_clip_labels = clip.tokenize(["a cow", "a rock", "soil or ground", "a horse", "a sheep", "a person", "a vehicle"]).to("cuda" if torch.cuda.is_available() else "cpu")
print("[BİLGİ] CLIP hazır.")


CLIP_LABEL_IDX = {"cow": 0, "sheep": 4}  # _clip_labels listesindeki indeksler

def verify_with_clip(frame, box, cls_name="cow"):
    """Crop'u CLIP ile kontrol et — hedef hayvan mı değil mi."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    pad = 20
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(w, x2+pad), min(h, y2+pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    # BGR → RGB → PIL
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = _clip_preprocess(pil_img).unsqueeze(0).to(device)
    label_list = ["cow", "rock", "soil", "horse", "sheep", "person", "vehicle"]
    with torch.no_grad():
        logits, _ = _clip_model(image_input, _clip_labels)
        probs = logits.softmax(dim=-1)[0]
    target_idx  = CLIP_LABEL_IDX.get(cls_name, 0)
    target_prob = probs[target_idx].item()
    print(f"[CLIP] {cls_name}={target_prob:.2f} | best={label_list[probs.argmax()]}")
    return target_prob > CLIP_THRESH

FARM_ANIMALS = {"cow", "sheep"}  # inek ve koyun takip ediliyor

# OIV7 modelinin sınıf isimlerini bizim standart isimlerimize eşle
OIV7_CLASS_MAP = {
    "cattle":  "cow",
    "bull":    "cow",
    "cow":     "cow",
    "sheep":   "sheep",
    "goat":    "goat",
    "horse":   "horse",
    "dog":     "dog",
    "cat":     "cat",
    "bird":    "bird",
    "chicken": "bird",
    "duck":    "bird",
    "turkey":  "bird",
}


TRAIL_LENGTH = 40  # Her hayvan için saklanacak maksimum pozisyon sayısı

CLASS_COLORS = {
    "cow":   (0, 200, 255),
    "sheep": (0, 255, 100),
    "horse": (255, 150, 0),
    "dog":   (200, 0, 255),
    "cat":   (255, 0, 150),
    "bird":  (0, 255, 255),
}
DEFAULT_COLOR = (200, 200, 200)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fit_to_screen(frame, max_w=1280, max_h=720):
    """Frame'i ekrana sığacak şekilde oranını koruyarak küçültür."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)  # 1.0: büyütme yok, sadece küçült
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def draw_trail(frame, trail_points, color):
    """Hayvanın geçmiş pozisyonlarını soluklaşan bir iz olarak çizer."""
    points = list(trail_points)
    for i in range(1, len(points)):
        alpha = i / len(points)  # eskiye doğru gittikçe 0'a yaklaşır
        thickness = max(1, int(3 * alpha))
        faded_color = tuple(int(c * alpha) for c in color)
        cv2.line(frame, points[i - 1], points[i], faded_color, thickness, cv2.LINE_AA)


def draw_info_panel(frame, counts, fps, total_unique):
    """Sol üst köşeye tür bazlı sayım paneli çizer."""
    px, py = 10, 10
    line_h = 26
    panel_h = 40 + line_h * (len(counts) + 2)
    panel_w = 220

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (120, 120, 120), 1)

    y = py + 24
    cv2.putText(frame, f"FPS: {fps:.1f}", (px + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    y += line_h
    cv2.putText(frame, f"Anlik Tespit: {sum(counts.values())}", (px + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 100), 1, cv2.LINE_AA)
    y += line_h + 4
    cv2.line(frame, (px + 8, y - 8), (px + panel_w - 8, y - 8), (80, 80, 80), 1)

    for cls_name, cnt in sorted(counts.items()):
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        text = f"{cls_name.capitalize():<10} {cnt:>3} adet"
        cv2.putText(frame, text, (px + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += line_h


def process_image(model, input_path, output_path, conf, target_classes):
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"[HATA] Gorsel okunamadi: {input_path}")
        return

    results = model(frame, conf=conf, verbose=False)
    frame_counts = defaultdict(int)

    for r in results:
        for box in r.boxes:
            conf_val = float(box.conf[0])
            if conf_val < CONF_LOW:
                continue

            raw_name = r.names[int(box.cls)].lower()
            cls_name = OIV7_CLASS_MAP.get(raw_name, raw_name)
            if cls_name not in target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if conf_val >= CONF_HIGH:
                is_cow = True
            else:
                is_cow = verify_with_clip(frame, (x1, y1, x2, y2), cls_name)

            if not is_cow:
                continue

            frame_counts[cls_name] += 1
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name.capitalize()} {conf_val:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            lx, ly = x1, max(y1 - lh - 8, 0)
            cv2.rectangle(frame, (lx, ly), (lx + lw + 6, ly + lh + 6), color, -1)
            cv2.putText(frame, label, (lx + 3, ly + lh + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    draw_info_panel(frame, frame_counts, 0.0, sum(frame_counts.values()))
    cv2.imwrite(output_path, frame)

    print("\n" + "=" * 45)
    print("  GORSEL ISLEME TAMAMLANDI")
    print("=" * 45)
    if frame_counts:
        for cls_name, cnt in sorted(frame_counts.items()):
            print(f"  {cls_name.capitalize():<12}: {cnt} adet")
    else:
        print("  Hicbir hayvan tespit edilemedi.")
    print(f"\n  Cikis gorseli: {output_path}")
    print("=" * 45)


def main():
    parser = argparse.ArgumentParser(description="Hayvan Tespit ve Sayım Sistemi")
    parser.add_argument("--input",       default="videos/input_video.mp4")
    parser.add_argument("--output",      default="outputs/result.mp4")
    parser.add_argument("--model",       default="yolov8s-oiv7.pt")
    parser.add_argument("--conf",        type=float, default=0.40)  # 0.40: daha fazla tespit
    parser.add_argument("--classes",     nargs="+", default=None)
    parser.add_argument("--no-show",     action="store_true")
    parser.add_argument("--display-w",   type=int, default=1280, help="Ekran gösterimi maks genişlik (piksel)")
    parser.add_argument("--display-h",   type=int, default=720,  help="Ekran gösterimi maks yükseklik (piksel)")
    parser.add_argument("--skip-frames", type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[HATA] Video bulunamadı: {args.input}")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    target_classes = set(c.lower() for c in args.classes) if args.classes else FARM_ANIMALS

    show = not args.no_show
    if show:
        try:
            test = np.zeros((4, 4, 3), dtype=np.uint8)
            cv2.imshow("__test__", test)
            cv2.waitKey(1)
            cv2.destroyWindow("__test__")
        except Exception:
            show = False
            print("[UYARI] Ekran desteği yok, sadece video kaydediliyor.")

    print(f"[BİLGİ] Model yükleniyor: {args.model}")
    model = YOLO(args.model)
    # GPU aktif — imgsz 640 ile tam kalite
    print("[BİLGİ] Model hazır.")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {args.input}")
        return

    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[BİLGİ] Video       : {frame_w}x{frame_h} @ {fps_video:.1f} FPS, {total_frames} kare")
    print(f"[BİLGİ] Güven eşiği : {args.conf} (altı sayılmaz)")

    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.output, fourcc, fps_video, (frame_w, frame_h))

    id_to_class  = {}
    id_to_trail  = defaultdict(lambda: [])
    id_verified  = {}   # track_id → {"result": bool, "last_frame": int}
    id_rejected  = set()  # kesin reddedilmiş ID'ler
    max_counts   = defaultdict(int)  # tür bazlı maksimum anlık tespit

    fps_counter = 0
    fps_display = 0.0
    fps_timer   = time.time()
    frame_idx   = 0

    print("[BİLGİ] İşlem başlıyor..." + (" Çıkmak için 'q'." if show else ""))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % args.skip_frames != 0:
            out_writer.write(frame)
            continue

        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            iou=0.60,  # yüksek IOU: üst üste binen yanlış kutuları eler
            tracker="bytetrack.yaml",
            verbose=False,
            imgsz=640,
            half=True,   # GPU'da FP16 — 2x hız, aynı kalite
        )

        frame_counts = defaultdict(int)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            confs   = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            names   = results[0].names

            for box, track_id, conf, cls_id in zip(boxes, ids, confs, classes):
                if conf < CONF_LOW:
                    continue

                raw_name = names[cls_id].lower()
                cls_name = OIV7_CLASS_MAP.get(raw_name, raw_name)
                if cls_name not in target_classes:
                    continue

                # --- İki aşamalı doğrulama ---
                if conf >= CONF_HIGH:
                    # Yüksek güven → direkt kabul
                    is_cow = True
                elif track_id in id_rejected:
                    continue
                else:
                    # Orta güven → cache kontrol, gerekirse CLIP'e sor
                    cache = id_verified.get(track_id)
                    if cache and (frame_idx - cache["last_frame"]) < VERIFY_CACHE_FRAMES:
                        is_cow = cache["result"]
                    else:
                        print(f"[CLIP] ID {track_id} doğrulanıyor (conf={conf:.2f})...")
                        is_cow = verify_with_clip(frame, box, cls_name)
                        id_verified[track_id] = {"result": is_cow, "last_frame": frame_idx}
                        if not is_cow:
                            id_rejected.add(track_id)
                            print(f"[CLIP] ID {track_id} reddedildi ({cls_name} değil)")

                if not is_cow:
                    continue

                id_to_class[track_id] = cls_name
                frame_counts[cls_name] += 1

                x1, y1, x2, y2 = map(int, box)
                color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

                # Hareket izi: hayvanın merkez noktasını listeye ekle
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trail = id_to_trail[track_id]
                trail.append((cx, cy))
                if len(trail) > TRAIL_LENGTH:
                    trail.pop(0)

                # İzi çiz
                draw_trail(frame, trail, color)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{cls_name.capitalize()} #{track_id}  {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                lx, ly = x1, max(y1 - lh - 8, 0)
                cv2.rectangle(frame, (lx, ly), (lx + lw + 6, ly + lh + 6), color, -1)
                cv2.putText(frame, label, (lx + 3, ly + lh + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = time.time()

        for cls, cnt in frame_counts.items():
            if cnt > max_counts[cls]:
                max_counts[cls] = cnt

        draw_info_panel(frame, frame_counts, fps_display, len(id_to_class))

        if total_frames > 0:
            bar_w = int(frame_w * frame_idx / total_frames)
            cv2.rectangle(frame, (0, frame_h - 4), (bar_w, frame_h), (0, 200, 100), -1)

        out_writer.write(frame)

        if show:
            try:
                display_frame = fit_to_screen(frame, args.display_w, args.display_h)
                cv2.imshow("Hayvan Sayim Sistemi  |  Cikis: Q", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[BİLGİ] Durduruldu.")
                    break
            except Exception as e:
                print(f"[UYARI] imshow hatası: {e}")
                show = False

        if frame_idx % 100 == 0:
            pct = f"{frame_idx/total_frames*100:.1f}%" if total_frames else str(frame_idx)
            aktif = ", ".join(f"{k}:{v}" for k, v in sorted(frame_counts.items())) or "—"
            print(f"  [{pct}] Aktif: {aktif} | FPS: {fps_display:.1f}")

    cap.release()
    out_writer.release()
    if show:
        cv2.destroyAllWindows()

    print("\n" + "=" * 45)
    print("  İŞLEM TAMAMLANDI — HAYVAN SAYIMI")
    print("=" * 45)
    if max_counts:
        for cls_name, cnt in sorted(max_counts.items()):
            print(f"  {cls_name.capitalize():<12}: {cnt} adet (maks anlık)")
        print(f"  {'Toplam':<12}: {sum(max_counts.values())} adet")
    else:
        print("  Hiçbir hayvan tespit edilemedi.")
        print("  → Güven eşiğini düşürmeyi deneyin: --conf 0.40")
    print(f"\n  Çıkış videosu: {args.output}")
    print("=" * 45)


if __name__ == "__main__":
    main()
