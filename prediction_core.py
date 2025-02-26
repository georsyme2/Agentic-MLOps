from datetime import datetime
from initialization import initialize_model, initialize_agent
from image_utils import preprocess_input_image, compute_image_metrics, augment_image
from model_utils import get_predictions
from agent_utils import analyze_prediction_with_agent
import os
import json
import numpy as np

# ====================== CONFIGURATION ======================

# Model initialization
MODEL_PATH = "best_model.h5"  # Replace with the actual path to your model
model = initialize_model(MODEL_PATH)

# Agent initialization
# Replace with your actual Hugging Face API key
agent = initialize_agent(API_KEY)

# Paths to log files
PREDICTION_MEMORY_FILE = "prediction_memory.json"
IMAGE_METRICS_LOG_FILE = "image_metrics_log.json"
EVALUATION_METRICS_LOG_FILE = "evaluation_metrics_log.json"

# ====================== HELPER FUNCTIONS ======================

def generate_prediction_id():
    """Generate a unique ID using the current timestamp."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


# ====================== PREDICTION PIPELINE ======================

def prediction_pipeline(image_path, sex, age_approx, anatom_site):
    """
    Full pipeline for predicting a single image, computing image metrics,
    checking robustness, evaluating via an agent, and saving results.
    """
    # Step 1: Preprocess the image
    img = preprocess_input_image(image_path)

    # Step 2: Compute image metrics
    image_metrics = compute_image_metrics(img)
    
    # Convert any NumPy arrays in image_metrics to Python lists
    image_metrics = convert_numpy_to_python(image_metrics)

    # Step 3: Get the model's prediction
    predicted_label, confidence_scores = get_predictions(model, img)

    # Add debug print statements to understand the structure
    print("DEBUG - predicted_label type:", type(predicted_label))
    print("DEBUG - predicted_label value:", predicted_label)
    print("DEBUG - confidence_scores type:", type(confidence_scores))
    print("DEBUG - confidence_scores value:", confidence_scores)

    # Convert NumPy values to Python native types
    if isinstance(predicted_label, np.ndarray):
        predicted_label = predicted_label.item()  # Convert scalar ndarray to Python scalar

    if isinstance(confidence_scores, np.ndarray):
        confidence_scores = confidence_scores.tolist()  # Convert array to list

    # More detailed extraction of confidence score
    if isinstance(confidence_scores, list):
        if len(confidence_scores) == 1 and not isinstance(confidence_scores[0], (list, np.ndarray)):
            # Simple list with one value
            confidence_score = confidence_scores[0]
        elif isinstance(confidence_scores[0], (list, np.ndarray)):
            # Nested list/array (common in softmax outputs)
            if isinstance(predicted_label, int) and predicted_label < len(confidence_scores[0]):
                confidence_score = confidence_scores[0][predicted_label]
            else:
                # If we can't match the index, take the highest value
                confidence_score = max(confidence_scores[0])
                print("DEBUG - Using max confidence:", confidence_score)
        else:
            # Multiple classes, take the one corresponding to predicted_label
            if isinstance(predicted_label, int) and predicted_label < len(confidence_scores):
                confidence_score = confidence_scores[predicted_label]
            else:
                # Fallback to max value
                confidence_score = max(confidence_scores)
                print("DEBUG - Using max confidence (fallback):", confidence_score)
    else:
        # If it's already a single value, use it directly
        confidence_score = confidence_scores

    print("DEBUG - Final confidence_score:", confidence_score)

    # Ensure confidence_score is a number, not a list or array
    if isinstance(confidence_score, list) or isinstance(confidence_score, np.ndarray):
        confidence_score = confidence_score[0] if len(confidence_score) > 0 else 0.0

    # If confidence_score is a NumPy type, convert it
    if isinstance(confidence_score, np.number):
        confidence_score = confidence_score.item()

    # Step 4: Perform robustness check
    robustness_status = "Stable"  # Placeholder for actual robustness check logic

    # Step 5: Analyze prediction with agent
    agent_decision = analyze_prediction_with_agent(agent, image_metrics, confidence_score, robustness_status)

    # Step 6: Save results to prediction memory
    prediction_id = generate_prediction_id()
    result = {
        "id": prediction_id,
        "image_metrics": image_metrics,
        "metadata": {
            "sex": sex,
            "age_approx": age_approx,
            "anatom_site": anatom_site
        },
        "prediction": predicted_label,
        "confidence_score": confidence_score,
        "robustness": robustness_status,
        "agent_decision": agent_decision
    }

    try:
        # Try to load existing prediction memory
        with open(PREDICTION_MEMORY_FILE, "r") as f:
            prediction_memory = json.load(f)
            
            # Ensure prediction_memory is a list
            if not isinstance(prediction_memory, list):
                prediction_memory = []
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or has invalid JSON, start with empty list
        prediction_memory = []

    prediction_memory.append(result)

    with open(PREDICTION_MEMORY_FILE, "w") as f:
        json.dump(prediction_memory, f, indent=2)

    return result

# Add this helper function to convert NumPy types to Python native types
def convert_numpy_to_python(obj):
    """
    Recursively convert NumPy types to Python native types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    return obj

# ====================== INITIALIZATION ======================

# Create `prediction_memory.json` if it doesn't exist
if not os.path.exists(PREDICTION_MEMORY_FILE):
    with open(PREDICTION_MEMORY_FILE, "w") as f:
        json.dump([], f)

print("✅ Prediction memory initialized.")

# ====================== EXAMPLE USAGE ======================
# Uncomment and replace "test_image.jpg" with an actual image path
# prediction_pipeline("test_image.jpg", "male", 45, "back")