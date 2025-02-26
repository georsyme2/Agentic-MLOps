import tensorflow as tf
import numpy as np

def get_predictions(model, img):
    """
    Get model predictions and confidence scores.

    Args:
        model (tensorflow.keras.models.Model): The trained Keras model.
        img (numpy.ndarray): Preprocessed image as a NumPy array.

    Returns:
        tuple: Predicted labels and confidence scores.
    """
    predictions = model.predict(img)

    # Check if the model is binary classification (sigmoid) or multi-class (softmax)
    if predictions.shape[1] == 1:  # Binary classification (sigmoid)
        confidence_scores = predictions  # Sigmoid outputs probability of class 1
        predicted_labels = (confidence_scores > 0.5).astype(int).flatten()  # Convert to 0 or 1
    else:  # Multi-class classification (softmax)
        confidence_scores = tf.nn.softmax(predictions, axis=1).numpy()  # Get softmax probabilities
        predicted_labels = np.argmax(confidence_scores, axis=1)  # Get predicted class index

    return predicted_labels, confidence_scores
