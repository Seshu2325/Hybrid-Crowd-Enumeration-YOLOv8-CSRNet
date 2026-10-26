"""
Visualization utilities for displaying results
"""

import cv2
import numpy as np
import config


def create_info_panel(
    image,
    yolo_count=None,
    csrnet_count=None,
    active_model=None,
    yolo_fps=None,
    csrnet_fps=None,
    density_level=None,
    mode='hybrid'
):
    """
    Create information overlay on image
    
    Args:
        image: Input image
        yolo_count: YOLO count
        csrnet_count: CSRNet count
        active_model: Active model name ('yolo' or 'csrnet')
        yolo_fps: YOLO FPS
        csrnet_fps: CSRNet FPS
        density_level: Density classification
        mode: Display mode ('hybrid' or 'adaptive')
    
    Returns:
        Image with info overlay
    """
    display = image.copy()
    h, w = display.shape[:2]
    
    # Create semi-transparent panel
    panel_height = 150
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
    
    # Text parameters
    font = config.FONT
    font_scale = config.FONT_SCALE
    thickness = config.FONT_THICKNESS
    
    y = 30
    
    # Title
    title = "ADAPTIVE HYBRID SYSTEM" if mode == 'adaptive' else "HYBRID COUNTING SYSTEM"
    cv2.putText(display, title, (10, y),
               font, font_scale + 0.2, (0, 255, 255), thickness)
    y += 35
    
    # Mode-specific display
    if mode == 'adaptive':
        # Adaptive mode: show active model
        if active_model:
            model_text = f"Active: {active_model.upper()}"
            model_color = (0, 255, 0) if active_model == 'yolo' else (0, 100, 255)
            cv2.putText(display, model_text, (10, y),
                       font, font_scale, model_color, thickness)
            y += 30
        
        # Show count from active model
        # Show count and FPS for the active model
        if active_model == 'yolo' and yolo_count is not None:
            count_text = f"Count: {int(yolo_count)} people"
            fps_text = (
                f"FPS: {yolo_fps:.1f}"
                if yolo_fps is not None else ""
            )

        elif active_model == 'csrnet' and csrnet_count is not None:
            count_text = f"Count: {csrnet_count:.0f} people"
            fps_text = (
                f"FPS: {csrnet_fps:.1f}"
                if csrnet_fps is not None else ""
            )

        else:
            count_text = ""
            fps_text = ""

        if count_text:
            cv2.putText(
                display, count_text, (10, y),
                font, font_scale, (255, 255, 255), thickness
            )

            if fps_text:
                cv2.putText(
                    display, fps_text, (300, y),
                    font, font_scale, (255, 255, 255), thickness
                )
    # Density level
    if density_level:
        y += 30
        density_colors = {
            'Sparse': (0, 255, 0),
            'Medium': (0, 165, 255),
            'Dense': (0, 0, 255)
        }
        color = density_colors.get(density_level, (255, 255, 255))
        cv2.putText(display, f"Density: {density_level}", (10, y),
                   font, font_scale, color, thickness)
    
    return display


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image
        boxes: List of boxes [x1, y1, x2, y2]
        color: Box color
        thickness: Line thickness
    
    Returns:
        Image with boxes
    """
    img = image.copy()
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    return img


def overlay_density_map(image, density_map, alpha=None):
    """
    Overlay density map on image
    
    Args:
        image: Original image
        density_map: Density map (numpy array)
        alpha: Transparency (defaults to config.OVERLAY_ALPHA)
    
    Returns:
        Image with density overlay
    """
    if alpha is None:
        alpha = config.OVERLAY_ALPHA
    
    if density_map is None or not isinstance(density_map, np.ndarray):
        return image
    
    if density_map.size == 0:
        return image
    
    # Resize density map to match image
    h, w = image.shape[:2]
    density_resized = cv2.resize(
        density_map.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Normalize
    d_min = density_resized.min()
    d_max = density_resized.max()
    
    if d_max > d_min:
        density_normalized = (density_resized - d_min) / (d_max - d_min)
    else:
        density_normalized = np.zeros_like(density_resized)
    
    # Apply colormap
    density_colored = (density_normalized * 255).astype(np.uint8)
    density_colored = cv2.applyColorMap(density_colored, cv2.COLORMAP_JET)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, density_colored, alpha, 0)
    
    return overlay


def create_visualization(
    image,
    result,
    mode='adaptive',
    show_boxes=True,
    show_density=True
):
    """Create a visualization with bounding boxes, density map and info panel."""
    display = image.copy()

    # Overlay density map, if available.
    if show_density and result.get('density_map') is not None:
        display = overlay_density_map(display, result['density_map'])

    # Draw YOLO bounding boxes, if available.
    boxes = result.get('boxes') or []
    if show_boxes and len(boxes) > 0:
        display = draw_boxes(display, boxes)

    if mode == 'adaptive':
        count = result.get('count', 0)
        if count < config.DENSITY_LOW:
            density_level = 'Sparse'
        elif count < config.DENSITY_MEDIUM:
            density_level = 'Medium'
        else:
            density_level = 'Dense'

        active_model = result.get('active_model')
        yolo_count = result.get('yolo_count')
        csrnet_count = result.get('csrnet_count')
        yolo_fps = result.get('yolo_fps')
        csrnet_fps = result.get('csrnet_fps')

        # Adaptive results store the active model's prediction in "count".
        if active_model == 'yolo':
            yolo_count = result.get('count', yolo_count)
            yolo_fps = result.get('fps', yolo_fps)
        elif active_model == 'csrnet':
            csrnet_count = result.get('count', csrnet_count)
            csrnet_fps = result.get('fps', csrnet_fps)

        display = create_info_panel(
            display,
            yolo_count=yolo_count,
            csrnet_count=csrnet_count,
            active_model=active_model,
            yolo_fps=yolo_fps,
            csrnet_fps=csrnet_fps,
            density_level=density_level,
            mode='adaptive'
        )
    else:
        # Basic hybrid mode displays the counts from both models.
        yolo_count = result.get('yolo_count', 0)
        if yolo_count < config.DENSITY_LOW:
            density_level = 'Sparse'
        elif yolo_count < config.DENSITY_MEDIUM:
            density_level = 'Medium'
        else:
            density_level = 'Dense'

        display = create_info_panel(
            display,
            yolo_count=result.get('yolo_count'),
            csrnet_count=result.get('csrnet_count'),
            active_model=None,
            yolo_fps=result.get('yolo_fps'),
            csrnet_fps=result.get('csrnet_fps'),
            density_level=density_level,
            mode='basic_hybrid'
        )

    return display
