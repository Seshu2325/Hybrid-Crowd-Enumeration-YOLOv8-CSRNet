"""
Adaptive Hybrid System - Overlap-Based Switching
Uses bounding box overlap ratio to detect congestion
"""

import config
import cv2
import numpy as np


class AdaptiveHybrid:

    def __init__(self, yolo_counter, csrnet_counter):
        self.yolo = yolo_counter
        self.csrnet = csrnet_counter

        # Overlap threshold (tune between 0.15 – 0.35)
        self.overlap_threshold = 0.25

        print("✓ Adaptive Hybrid Initialized")
        print(f"CSRNet activates when overlap ratio > {self.overlap_threshold}")

    # ---------------------------------------------------------
    def predict_basic_hybrid(self, image):

        yolo_result = self.yolo.predict(image, return_boxes=True)
        csrnet_result = self.csrnet.predict(image, return_density_map=True)

        return {
            "mode": "basic_hybrid",
            "active_model": "both",
            "count": (yolo_result["count"] + csrnet_result["count"]) / 2,
            "yolo_count": yolo_result["count"],
            "csrnet_count": csrnet_result["count"],
            "boxes": yolo_result.get("boxes", []),
            "density_map": csrnet_result.get("density_map"),
            "fps": min(yolo_result["fps"], csrnet_result["fps"]),
            "decision_reason": "Both models executed"
        }

    # ---------------------------------------------------------

    def predict_adaptive(self, image):

    # Step 1 — Run YOLO
        yolo_result = self.yolo.predict(image, return_boxes=True)
        yolo_count = yolo_result['count']
        boxes = yolo_result['boxes']

        image_area = image.shape[0] * image.shape[1]

        # -------------------------------------------------
        # Signal 1: Average bounding box size
        # -------------------------------------------------
        avg_area = 0
        if boxes:
            areas = [(x2-x1)*(y2-y1) for (x1,y1,x2,y2) in boxes]
            avg_area = sum(areas) / len(areas)

        box_density_ratio = avg_area / image_area if image_area > 0 else 0

        # -------------------------------------------------
        # Signal 2: Edge density (crowd texture detector)
        # -------------------------------------------------
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.sum(edges > 0) / image_area

        # -------------------------------------------------
        # Decision Logic
        # -------------------------------------------------

        # Case A: Clearly sparse
        if yolo_count > 20 and box_density_ratio > 0.02:
            return {
                'count': yolo_count,
                'active_model': 'yolo',
                'fps': yolo_result['fps'],
                'boxes': boxes,
                'density_map': None,
                'mode': 'adaptive'
            }

        # Case B: Suspicious undercount (dense texture but few detections)
        if yolo_count < 20 and edge_ratio > 0.08:
            csr_result = self.csrnet.predict(image, return_density_map=True)

            return {
                'count': csr_result['count'],
                'active_model': 'csrnet',
                'fps': csr_result['fps'],
                'boxes': boxes,
                'density_map': csr_result.get('density_map'),
                'mode': 'adaptive'
            }

        # Case C: Medium ambiguous → run CSR for safety
        if edge_ratio > 0.12:
            csr_result = self.csrnet.predict(image, return_density_map=True)

            return {
                'count': csr_result['count'],
                'active_model': 'csrnet',
                'fps': csr_result['fps'],
                'boxes': boxes,
                'density_map': csr_result.get('density_map'),
                'mode': 'adaptive'
            }

        # Default fallback → YOLO
        return {
            'count': yolo_count,
            'active_model': 'yolo',
            'fps': yolo_result['fps'],
            'boxes': boxes,
            'density_map': None,
            'mode': 'adaptive'
        }
    # ---------------------------------------------------------
    def _compute_overlap_ratio(self, boxes):

        if len(boxes) < 2:
            return 0.0

        total_overlap = 0
        total_area = 0

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            area_i = (x2 - x1) * (y2 - y1)
            total_area += area_i

            for j in range(i + 1, len(boxes)):
                xa1, ya1, xa2, ya2 = boxes[j]

                inter_x1 = max(x1, xa1)
                inter_y1 = max(y1, ya1)
                inter_x2 = min(x2, xa2)
                inter_y2 = min(y2, ya2)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    total_overlap += (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        if total_area == 0:
            return 0.0

        return total_overlap / total_area


def create_adaptive_hybrid(yolo_counter, csrnet_counter):
    return AdaptiveHybrid(yolo_counter, csrnet_counter)