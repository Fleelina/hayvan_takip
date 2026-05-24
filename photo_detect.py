import argparse
import os
from collections import defaultdict

import cv2
import torch
import clip
from PIL import Image
from ultralytics import YOLO

CONF_HIGH = 0.75
CLIP_THRESH = 0.35

CLASS_COLORS = {
    "cow":   (0, 200, 255),
    "sheep": (0, 255, 100),
    "horse": (255, 150, 0),
    "dog":   (200, 0, 255),
    "cat":   (255, 0, 150),
    "bird":  (0, 255, 255),
}
DEFAULT_COLOR = (200, 200, 200)

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

print("[BILGI] CLIP yukleniyor...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
CLIP_LABELS = clip.tokenize(
    ["a cow", "a rock", "soil or ground", "a horse", "a sheep", "a person", "a vehicle"]
).to(DEVICE)
print("[BILGI] CLIP hazir.")


def verify_with_clip(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    pad = 20
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    image_input = CLIP_PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, _ = CLIP_MODEL(image_input, CLIP_LABELS)
        probs = logits.softmax(dim=-1)[0]

    cow_prob = probs[0].item()
    print(f"[CLIP] cow={cow_prob:.2f}")
    return cow_prob > CLIP_THRESH


def draw_info_panel(frame, counts):
    px, py = 10, 10
    line_h = 26
    panel_h = 40 + line_h * (len(counts) + 1)
    panel_w = 220

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (120, 120, 120), 1)

    y = py + 24
    cv2.putText(
        frame,
        f"Anlik Tespit: {sum(counts.values())}",
        (px + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 220, 100),
        1,
        cv2.LINE_AA,
    )
    y += line_h + 4
    cv2.line(frame, (px + 8, y - 8), (px + panel_w - 8, y - 8), (80, 80, 80), 1)

    for cls_name, cnt in sorted(counts.items()):
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        text = f"{cls_name.capitalize():<10} {cnt:>3} adet"
        cv2.putText(frame, text, (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += line_h


def main():
    parser = argparse.ArgumentParser(description="Tek gorselde hayvan tespiti")
    parser.add_argument("--input", required=True, help="Ornek: photos/cow_photo.jpg")
    parser.add_argument("--output", default="outputs/cow_photo_result.jpg")
    parser.add_argument("--model", default="yolov8s-oiv7.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[HATA] Gorsel bulunamadi: {args.input}")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    frame = cv2.imread(args.input)
    if frame is None:
        print(f"[HATA] Gorsel okunamadi: {args.input}")
        return

    print(f"[BILGI] Model yukleniyor: {args.model}")
    model = YOLO(args.model)
    results = model(frame, conf=args.conf, verbose=False)

    frame_counts = defaultdict(int)

    for r in results:
        for box in r.boxes:
            raw_name = r.names[int(box.cls)].lower()
            cls_name = OIV7_CLASS_MAP.get(raw_name, raw_name)
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_name == "cow" and conf_val < CONF_HIGH:
                if not verify_with_clip(frame, (x1, y1, x2, y2)):
                    continue

            frame_counts[cls_name] += 1
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf_val:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            lx, ly = x1, max(y1 - lh - 8, 0)
            cv2.rectangle(frame, (lx, ly), (lx + lw + 6, ly + lh + 6), color, -1)
            cv2.putText(frame, label, (lx + 3, ly + lh + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    draw_info_panel(frame, frame_counts)
    cv2.imwrite(args.output, frame)

    print("\nSonuc:")
    if frame_counts:
        for cls_name, cnt in sorted(frame_counts.items()):
            print(f"- {cls_name}: {cnt}")
    else:
        print("- Hicbir sey tespit edilemedi")
    print(f"Kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
