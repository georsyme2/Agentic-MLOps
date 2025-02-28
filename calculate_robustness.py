# calculate_robustness.py
import numpy as np

def calculate_robustness(model, original_img, original_prediction, num_augmentations=5):
    """
    Calculate robustness by applying multiple augmentations and checking if predictions remain stable.
    
    Args:
        model: The trained model
        original_img: The preprocessed original image
        original_prediction: The prediction made on the original image
        num_augmentations: Number of augmentations to apply
        
    Returns:
        tuple: (robustness_status, stability_score, prediction_counts)
            - robustness_status: "Stable" or "Unstable"
            - stability_score: Percentage of predictions that matched the original
            - prediction_counts: Dictionary with counts of each prediction
    """
    from image_utils import augment_image
    from model_utils import get_predictions
    
    # Keep track of predictions
    predictions = []
    confidence_scores = []
    
    # Make a copy of the original image to avoid modifying it
    img_copy = np.copy(original_img)
    
    # Augment the image multiple times and predict
    for i in range(num_augmentations):
        try:
            # Apply augmentation (noise and blur)
            augmented_img = augment_image(img_copy)
            
            # Ensure augmented image has correct shape for model
            if len(augmented_img.shape) != 4:
                augmented_img = np.expand_dims(augmented_img, axis=0)
            
            # Get prediction
            pred_label, conf_scores = get_predictions(model, augmented_img)
            
            # Convert to Python native types if needed
            if isinstance(pred_label, np.ndarray):
                pred_label = pred_label.item()
                
            predictions.append(pred_label)
            confidence_scores.append(conf_scores)
            
        except Exception as e:
            print(f"Error during augmentation {i}: {e}")
            # Continue with the next augmentation
            continue
    
    # If we couldn't get any predictions, return default values
    if not predictions:
        return "Unknown", 0.0, {str(original_prediction): 1}
    
    # Calculate stability metrics
    matching_count = predictions.count(original_prediction)
    stability_score = matching_count / len(predictions)
    
    # Count occurrences of each prediction
    prediction_counts = {}
    for pred in set(predictions + [original_prediction]):
        # Convert the prediction to string to ensure it can be serialized to JSON
        pred_str = str(pred)
        prediction_counts[pred_str] = predictions.count(pred)
    
    # Add original prediction if it's not in the augmented results
    if str(original_prediction) not in prediction_counts:
        prediction_counts[str(original_prediction)] = 0
    
    # Determine robustness status
    # You can adjust the threshold based on your requirements
    robustness_status = "Stable" if stability_score >= 0.7 else "Unstable"
    
    return robustness_status, stability_score, prediction_counts