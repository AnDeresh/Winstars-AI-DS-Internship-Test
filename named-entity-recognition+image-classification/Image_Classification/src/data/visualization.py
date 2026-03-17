import matplotlib.pyplot as plt
import os

def display_predictions(test_images, predict_fn, num_images=10):
    """
    Display predictions for a list of test images in a grid format.
    
    Args:
        test_images (list): List of paths to test images.
        predict_fn (callable): Function that takes an image path and returns (predicted_label, image).
        num_images (int): Number of images to display.
    """
    selected_images = test_images[:num_images]
    n_cols = 5
    n_rows = (len(selected_images) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx, image_path in enumerate(selected_images):
        predicted_label, image = predict_fn(image_path)
        axes[idx].imshow(image)
        axes[idx].set_title(f"Predicted: {predicted_label}", fontsize=12, fontweight="bold")
        axes[idx].axis("off")
    
    # Turn off any unused subplots
    for ax in axes[len(selected_images):]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
