"""
Metrics utilities for evaluating crowd counting performance
"""

import numpy as np


def mean_absolute_error(predictions, ground_truths):
    """
    Calculate Mean Absolute Error (MAE)
    
    Args:
        predictions: List of predicted counts
        ground_truths: List of ground truth counts
    
    Returns:
        float: MAE
    """
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    mae = np.mean(np.abs(predictions - ground_truths))
    return float(mae)


def mean_squared_error(predictions, ground_truths):
    """
    Calculate Mean Squared Error (MSE)
    
    Args:
        predictions: List of predicted counts
        ground_truths: List of ground truth counts
    
    Returns:
        float: MSE
    """
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    mse = np.mean((predictions - ground_truths) ** 2)
    return float(mse)


def root_mean_squared_error(predictions, ground_truths):
    """
    Calculate Root Mean Squared Error (RMSE)
    
    Args:
        predictions: List of predicted counts
        ground_truths: List of ground truth counts
    
    Returns:
        float: RMSE
    """
    mse = mean_squared_error(predictions, ground_truths)
    return float(np.sqrt(mse))


def compute_metrics(yolo_counts, csrnet_counts, ground_truths):
    """
    Compute all metrics for both models
    
    Args:
        yolo_counts: List of YOLO predictions
        csrnet_counts: List of CSRNet predictions
        ground_truths: List of ground truths
    
    Returns:
        dict: Metrics for both models
    """
    # YOLO metrics
    yolo_mae = mean_absolute_error(yolo_counts, ground_truths)
    yolo_mse = mean_squared_error(yolo_counts, ground_truths)
    yolo_rmse = root_mean_squared_error(yolo_counts, ground_truths)
    
    # CSRNet metrics
    csrnet_mae = mean_absolute_error(csrnet_counts, ground_truths)
    csrnet_mse = mean_squared_error(csrnet_counts, ground_truths)
    csrnet_rmse = root_mean_squared_error(csrnet_counts, ground_truths)
    
    return {
        'yolo': {
            'mae': yolo_mae,
            'mse': yolo_mse,
            'rmse': yolo_rmse
        },
        'csrnet': {
            'mae': csrnet_mae,
            'mse': csrnet_mse,
            'rmse': csrnet_rmse
        },
        'winner': 'YOLO' if yolo_mae < csrnet_mae else 'CSRNet'
    }


def print_metrics_summary(metrics, yolo_fps=None, csrnet_fps=None):
    """
    Print formatted metrics summary
    
    Args:
        metrics: Metrics dictionary from compute_metrics()
        yolo_fps: Optional YOLO FPS list
        csrnet_fps: Optional CSRNet FPS list
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print("\nYOLOv8:")
    print(f"  MAE:  {metrics['yolo']['mae']:.2f}")
    print(f"  MSE:  {metrics['yolo']['mse']:.2f}")
    print(f"  RMSE: {metrics['yolo']['rmse']:.2f}")
    if yolo_fps:
        print(f"  Avg FPS: {np.mean(yolo_fps):.2f}")
    
    print("\nCSRNet:")
    print(f"  MAE:  {metrics['csrnet']['mae']:.2f}")
    print(f"  MSE:  {metrics['csrnet']['mse']:.2f}")
    print(f"  RMSE: {metrics['csrnet']['rmse']:.2f}")
    if csrnet_fps:
        print(f"  Avg FPS: {np.mean(csrnet_fps):.2f}")
    
    print(f"\nBest Model (by MAE): {metrics['winner']}")
    print("="*60 + "\n")


def save_metrics(metrics, filepath):
    """
    Save metrics to file
    
    Args:
        metrics: Metrics dictionary
        filepath: Output file path
    """
    import json
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Metrics saved to {filepath}")


if __name__ == '__main__':
    # Test metrics
    predictions = [100, 200, 150]
    ground_truths = [105, 195, 160]
    
    mae = mean_absolute_error(predictions, ground_truths)
    mse = mean_squared_error(predictions, ground_truths)
    rmse = root_mean_squared_error(predictions, ground_truths)
    
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
