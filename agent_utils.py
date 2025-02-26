import json
import os
import time
from smolagents import ToolCallingAgent, HfApiModel, Tool
from promts import DECISION_PROMPT, IMAGE_CONTENT_PROMPT, PREDICTION_ANALYSIS_PROMPT, TECHNICAL_QUALITY_PROMPT

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


# agent_utils.py - Update these functions

def run_with_retry(agent, prompt, retries=3, delay=10):
    """
    Attempts to run the agent with retries if Hugging Face API is overloaded.
    For text responses, returns the text directly.
    For JSON responses, attempts to parse them.
    """
    for attempt in range(retries):
        try:
            response = agent.run(prompt)
            
            # If we expect JSON (for the final decision), try to parse it
            if "JSON format" in prompt:
                try:
                    # Try to find and extract JSON content if it exists
                    import re
                    json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        return json.loads(json_str)
                    return json.loads(response)
                except json.JSONDecodeError:
                    print(f"⚠️ JSON parsing error on attempt {attempt + 1}. Retrying...")
            else:
                # For text responses, just return the text
                return response
                
        except Exception as e:
            print(f"⚠️ API call failed on attempt {attempt + 1}: {e}. Retrying in {delay} seconds...")
        
        time.sleep(delay)  # Wait before retrying

    print("❌ All retry attempts failed. Returning fallback response.")
    
    # Provide fallback responses based on context
    # In the fallback response from run_with_retry:
    if "JSON format" in prompt:
        return {
            "assessment_summary": "Analysis could not be completed due to API limitations.",
            "image_feedback": "Image quality analysis unavailable.",
            "content_assessment": "Content analysis unavailable.",
            "prediction_feedback": "Prediction reliability assessment unavailable.",
            "user_suggestion": "Consider taking a clearer image under better conditions.",
            "override_decision": {"override": False, "corrected_prediction": None}
        }
    else:
        return "Analysis could not be completed due to API limitations."


def analyze_prediction_with_agent(agent, image_metrics, confidence_score, robustness_status, sex=None, age_approx=None, anatom_site=None):
    """
    Uses a modular approach to analyze predictions with separate analyses for:
    1. Technical image quality
    2. Image content assessment
    3. Prediction reliability
    4. Final decision synthesis
    """
    # Load reference logs
    training_image_metrics = load_json_log("image_metrics_log.json")
    evaluation_metrics = load_json_log("evaluation_metrics_log.json")

    if not training_image_metrics:
        training_image_metrics = {
            "sharpness": {"mean": 600, "median": 500},
            "brightness": {"mean": 120, "median": 120},
            "contrast": {"mean": 60, "median": 65},
            "fourier_descriptors": {"mean": [1.0, 0.25, 0.15, 0.06, 0.03, 0.04, 0.03, 0.02, 0.01, 0.01]}
        }
    
    if not evaluation_metrics:
        evaluation_metrics = {
            "categories": {
                "TP": {"confidence_scores": {"mean": 0.65}},
                "FP": {"confidence_scores": {"mean": 0.55}},
                "TN": {"confidence_scores": {"mean": 0.30}},
                "FN": {"confidence_scores": {"mean": 0.35}}
            }
        }

    # Step 1: Analyze technical image quality
    technical_quality_analysis = run_with_retry(
        agent,
        TECHNICAL_QUALITY_PROMPT.format(
            sharpness=image_metrics["sharpness"],
            brightness=image_metrics["brightness"],
            contrast=image_metrics["contrast"],
            training_sharpness_mean=training_image_metrics["sharpness"]["mean"],
            training_sharpness_median=training_image_metrics["sharpness"]["median"],
            training_brightness_mean=training_image_metrics["brightness"]["mean"],
            training_brightness_median=training_image_metrics["brightness"]["median"],
            training_contrast_mean=training_image_metrics["contrast"]["mean"],
            training_contrast_median=training_image_metrics["contrast"]["median"]
        )
    )
    
    # Step 2: Analyze image content
    content_analysis = run_with_retry(
        agent,
        IMAGE_CONTENT_PROMPT.format(
            fourier_descriptors=str(image_metrics["fourier_descriptors"]),  # Convert to string
            training_fourier_descriptors=str(training_image_metrics["fourier_descriptors"])  # Convert to string
        )
    )
    
    # Step 3: Analyze prediction reliability
    prediction_analysis = run_with_retry(
        agent,
        PREDICTION_ANALYSIS_PROMPT.format(
            confidence_score=confidence_score,
            robustness_status=robustness_status,
            anatom_site=anatom_site if anatom_site else "unknown",
            age_approx=age_approx if age_approx else "unknown",
            sex=sex if sex else "unknown",
            tp_confidence_mean=evaluation_metrics["categories"]["TP"]["confidence_scores"]["mean"],
            fp_confidence_mean=evaluation_metrics["categories"]["FP"]["confidence_scores"]["mean"],
            tn_confidence_mean=evaluation_metrics["categories"]["TN"]["confidence_scores"]["mean"],
            fn_confidence_mean=evaluation_metrics["categories"]["FN"]["confidence_scores"]["mean"]
        )
    )
    
    # Step 4: Make final decision
    final_decision = run_with_retry(
        agent,
        DECISION_PROMPT.format(
            technical_quality_analysis=technical_quality_analysis,
            content_analysis=content_analysis,
            prediction_analysis=prediction_analysis
        )
    )
    
    # If final_decision is a string instead of a dict (JSON parsing failed), create a simple dict
    if isinstance(final_decision, str):
        final_decision = {
            "assessment_summary": "Analysis completed with formatting issues.",
            "image_feedback": "Image quality analysis completed but results could not be properly formatted.",
            "prediction_feedback": "Prediction reliability assessment completed but results could not be properly formatted.",
            "user_suggestion": "Consider retrying with a clearer image.",
            "override_decision": {"override": False, "corrected_prediction": None}
        }
    
    # Store all analyses for possible display in the UI
    final_decision["detailed_analyses"] = {
        "technical_quality": technical_quality_analysis,
        "content": content_analysis,
        "reliability": prediction_analysis
    }
    
    return final_decision