import json
import os
import time
from smolagents import ToolCallingAgent, HfApiModel, Tool
from promts import IMAGE_QUALITY_PROMPT, PREDICTION_PROMPT, DECISION_PROMPT

# Tool: Collect Image Metrics
class ImageQualityAssessmentTool(Tool):
    name = "image_quality_assessment"
    description = "Collects image sharpness, brightness, contrast, and Fourier descriptors."
    inputs = {
        "image_metrics": {"type": "object", "description": "Computed image quality metrics in JSON format."}
    }
    output_type = "string"  # The model expects string output, even for JSON data

    def forward(self, image_metrics: dict) -> str:
        """Returns image quality metrics in JSON format."""
        return json.dumps(image_metrics)  # Convert to string to avoid serialization issues


class PredictionEvaluationTool(Tool):
    name = "prediction_data"
    description = "Collects prediction confidence and robustness status."
    inputs = {
        "confidence_score": {"type": "number", "description": "The confidence score of the prediction."},
        "robustness_status": {"type": "string", "description": "Indicates if the model's prediction was stable or changed."}
    }
    output_type = "string"

    def forward(self, confidence_score: float, robustness_status: str) -> str:
        """Returns prediction confidence and robustness status in JSON format."""
        prediction_data = {
            "confidence_score": confidence_score,
            "robustness_status": robustness_status
        }
        return json.dumps(prediction_data)  # Ensure output is a string


def load_json_log(file_path):
    """
    Loads a JSON log file and returns its contents.
    If the file does not exist or is empty, returns None.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                return data if data else None  # Ensure we don't return an empty dict
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse {file_path}. It may be corrupted.")
            return None
    print(f"⚠️ {file_path} not found.")
    return None


def run_with_retry(agent, prompt, retries=3, delay=10):
    """
    Attempts to run the agent with retries if Hugging Face API is overloaded.
    If the API fails multiple times, returns None instead of breaking the pipeline.
    """
    for attempt in range(retries):
        try:
            response = agent.run(prompt)
            return json.loads(response)  # Ensure valid JSON response
        except json.JSONDecodeError:
            print(f"⚠️ JSON parsing error on attempt {attempt + 1}. Retrying in {delay} seconds...")
        except Exception as e:
            print(f"⚠️ API call failed on attempt {attempt + 1}: {e}. Retrying in {delay} seconds...")
        time.sleep(delay)  # Wait before retrying

    print("❌ All retry attempts failed. Returning None.")
    return None  # If all retries fail, return None to prevent crashes

def analyze_prediction_with_agent(agent, image_metrics, confidence_score, robustness_status):
    """
    Uses the AI agent to analyze the prediction, image quality, and user reliability.
    Enhances decision-making by comparing with training set statistics.
    """
    # Load reference logs (You'll need to provide the actual paths)
    training_image_metrics = load_json_log("image_metrics_log.json")
    evaluation_metrics = load_json_log("evaluation_metrics_log.json")

    # Prepare the data for the prompts with json.dumps
    formatted_image_metrics = json.dumps(image_metrics, indent=2)
    formatted_training_image_metrics = json.dumps(training_image_metrics, indent=2) if training_image_metrics else "Training data not available."
    formatted_evaluation_metrics = json.dumps(evaluation_metrics, indent=2) if evaluation_metrics else "Evaluation data not available."

    # Ensure confidence_score is a float
    confidence_score = float(confidence_score)

    # Step 1: Compare Image Quality Against Training Set
    image_quality_response = run_with_retry(
        agent,
        IMAGE_QUALITY_PROMPT.format(
            image_metrics=formatted_image_metrics,  # Pass the formatted string
            training_image_metrics=formatted_training_image_metrics,
            evaluation_metrics=formatted_evaluation_metrics  # Pass the formatted string
        )
    )

    if not image_quality_response:
        image_quality_response = {
            "image_feedback": "Could not compare image to training set due to missing data or server timeout.",
            "issues_detected": [] # Assign an empty list
        }

    # Step 2: Compare Prediction Confidence Against Evaluation Metrics Log
    prediction_response = run_with_retry(
        agent,
        PREDICTION_PROMPT.format(
            confidence_score=confidence_score,
            robustness_status=robustness_status,
            evaluation_metrics=formatted_evaluation_metrics  # Pass the formatted string
        )
    )

    if not prediction_response:
        prediction_response = {
            "prediction_feedback": "Could not compare confidence to past evaluations due to missing data.",
            "trust_level": "Unknown"
        }

    # Step 3: Make Final Decision Using All Available Information
    final_decision = run_with_retry(
        agent,
        DECISION_PROMPT.format(
            image_quality_response=json.dumps(image_quality_response, indent=2),  # Convert dict to JSON string
            prediction_response=json.dumps(prediction_response, indent=2)  # Convert dict to JSON string
        )
    )

    if not final_decision:
        final_decision = {
            "image_feedback": "Final decision processing failed due to server timeout.",
            "prediction_feedback": "Could not determine the accuracy of the model's prediction.",
            "user_suggestion": "Consider taking a clearer image under better conditions.",
            "override_decision": {"override": False, "corrected_prediction": None}
        }

    return final_decision