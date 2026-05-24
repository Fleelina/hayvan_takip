"""
Kare Çıkarma Scripti — Fine-tuning için dataset hazırlığı
Kullanım: python extract_frames.py --input videos/input_video2.mp4
"""

import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Videodan kare çıkar")
    parser.add_argument("--input",    default="videos/input_video2.mp4")
    parser.add_argument("--output",   default="dataset/images")
    parser.add_argument("--every",    type=int, default=15,
                        help="Her kaç karede bir görsel alınsın (varsayılan: 15)")
    parser.add_argument("--max",      type=int, default=500,
                        help="Maksimum çıkarılacak kare sayısı (varsayılan: 500)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[HATA] Video bulunamadı: {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {args.input}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration     = total_frames / fps if fps else 0

    print(f"[BİLGİ] Video     : {args.input}")
    print(f"[BİLGİ] Süre      : {duration:.1f} saniye  ({total_frames} kare @ {fps:.1f} FPS)")
    print(f"[BİLGİ] Her {args.every}. kare alınacak, maksimum {args.max} görsel")
    print(f"[BİLGİ] Çıktı     : {args.output}/\n")

    saved   = 0
    f_idx   = 0

    while saved < args.max:
        ret, frame = cap.read()
        if not ret:
            break

        if f_idx % args.every == 0:
            filename = os.path.join(args.output, f"frame_{f_idx:06d}.jpg")
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            if saved % 50 == 0:
                print(f"  {saved} görsel kaydedildi...")

        f_idx += 1

    cap.release()

    print(f"\n✅ Tamamlandı: {saved} görsel → {args.output}/")
    print("\nSonraki adım:")
    print("  1. https://roboflow.com adresine git, ücretsiz hesap aç")
    print("  2. 'Create Project' → Object Detection → 'cow', 'sheep' vb. sınıf isimleri")
    print(f"  3. '{args.output}' klasöründeki görselleri yükle")
    print("  4. Her görselde hayvanların etrafına kutucuk çiz")
    print("  5. Export → YOLOv8 formatında indir")
    print("  6. Bana haber ver, train.py'yi yazayım!")

if __name__ == "__main__":
    main()
