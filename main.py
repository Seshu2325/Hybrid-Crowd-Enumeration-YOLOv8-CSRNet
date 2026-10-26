"""
Hybrid Crowd Counting System
Clean Display Version (No Blur / No Forced Zoom)
"""

import argparse
import sys
import os
import cv2
import torch
import ctypes

import config
from models import YOLOCounter, CSRNetCounter, AdaptiveHybrid
from utils import create_visualization


# ==========================================================
# Resize only if image is larger than screen
# ==========================================================
def resize_if_larger_than_screen(image):
    try:
        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
    except:
        # fallback resolution
        screen_w, screen_h = 1920, 1080

    h, w = image.shape[:2]

    if w > screen_w or h > screen_h:
        scale = min(screen_w / w, screen_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_AREA)

    return image


# ==========================================================
class HybridCrowdSystem:

    def __init__(self):
        print("\n" + "=" * 60)
        print("HYBRID CROWD COUNTING SYSTEM")
        print("=" * 60 + "\n")

        print("[1/2] Loading YOLO...")
        self.yolo = YOLOCounter()

        print("\n[2/2] Loading CSRNet...")
        self.csrnet = CSRNetCounter(config.CSRNET_WEIGHTS_PATH)

        print("\n[3/3] Creating Adaptive Hybrid...")
        self.hybrid = AdaptiveHybrid(self.yolo, self.csrnet)

        print("\n✓ SYSTEM READY\n")

    # ======================================================
    def display_image(self, vis, window_name):
        vis_display = resize_if_larger_than_screen(vis)

        # IMPORTANT: Use AUTOSIZE to prevent stretching
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window_name, vis_display)

    # ======================================================
    def process_image(self, image_path, mode="adaptive", save=False):

        print(f"Processing image: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            print("✗ Failed to load image")
            return

        # Detection on ORIGINAL image (no pre-resize)
        if mode == "adaptive":
            result = self.hybrid.predict_adaptive(image)
        else:
            result = self.hybrid.predict_basic_hybrid(image)

        vis = create_visualization(image, result, mode, True, True)

        print("\nResults:")
        print(f"Active Model: {result['active_model'].upper()}")
        print(f"Count: {result['count']:.1f}")
        print(f"FPS: {result['fps']:.1f}")

        if save:
            output_path = os.path.join(
                config.OUTPUT_DIR,
                f"result_{os.path.basename(image_path)}"
            )
            cv2.imwrite(output_path, vis)
            print(f"✓ Saved to {output_path}")

        self.display_image(vis, "Hybrid Crowd Counting")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ======================================================
    def process_video(self, video_path, mode="adaptive", save=False):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("✗ Failed to open video")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if save:
            output_path = os.path.join(
                config.OUTPUT_DIR,
                f"result_{os.path.basename(video_path)}"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc,
                                  fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if mode == "adaptive":
                result = self.hybrid.predict_adaptive(frame)
                print(f"Model: {result['active_model']} | Count: {result['count']:.1f}")
            else:
                result = self.hybrid.predict_basic_hybrid(frame)

            vis = create_visualization(frame, result,
                                       mode, True, True)

            if out:
                out.write(vis)

            self.display_image(vis,
                               "Hybrid Crowd Counting - Video")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if out:
            out.release()
            print("✓ Video saved")

        cv2.destroyAllWindows()

    # ======================================================
    def process_webcam(self, camera=0, mode="adaptive"):

        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            print("✗ Failed to open webcam")
            return

        print("Press Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if mode == "adaptive":
                result = self.hybrid.predict_adaptive(frame)
            else:
                result = self.hybrid.predict_basic_hybrid(frame)

            vis = create_visualization(frame, result,
                                       mode, True, True)

            self.display_image(vis,
                               "Hybrid Crowd Counting - Webcam")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ==========================================================
def main():

    parser = argparse.ArgumentParser(
        description="Hybrid Crowd Counting System"
    )

    parser.add_argument("--image", type=str)
    parser.add_argument("--video", type=str)
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mode", type=str,
                        default="adaptive",
                        choices=["adaptive", "basic_hybrid"])
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    config.validate_config()

    system = HybridCrowdSystem()

    if args.image:
        system.process_image(args.image,
                             mode=args.mode,
                             save=args.save)

    elif args.video:
        system.process_video(args.video,
                             mode=args.mode,
                             save=args.save)

    elif args.webcam:
        system.process_webcam(camera=args.camera,
                              mode=args.mode)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()