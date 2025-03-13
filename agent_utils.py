# agent_utils.py - Simplified for tool-based approach
import json
import os
# Format the unified prompt with all necessary information
from promts import UNIFIED_ANALYSIS_PROMPT
from agent_tools import latest_analysis_response

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

def create_fallback_response():
    """Create a fallback response when all retry attempts fail."""
    return {
        "assessment_summary": "Analysis could not be completed due to API limitations.",
        "image_feedback": "Image quality analysis unavailable.",
        "content_assessment": "Content analysis unavailable.",
        "prediction_feedback": "Prediction reliability assessment unavailable.",
        "user_suggestion": "Consider taking a clearer image under better conditions.",
        "override_decision": {"override": False, "corrected_prediction": None},
        "detailed_analyses": {
            "technical_quality": "Technical quality analysis unavailable.",
            "content": "Content analysis unavailable.",
            "reliability": "Reliability analysis unavailable."
        }
    }
def analyze_prediction_with_agent(agent, image_metrics, confidence_score, robustness_status, stability_score=1.0, sex=None, age_approx=None, anatom_site=None):
    """
    Uses a unified agent approach with tools to analyze predictions.
    """
    # Load reference logs
    training_image_metrics = load_json_log("image_metrics_log.json")
    evaluation_metrics = load_json_log("evaluation_metrics_log.json")


    # Fallback values if log files aren't found
    if not training_image_metrics:
        training_image_metrics = {
            "sharpness": {"mean": 600, "median": 500},
            "brightness": {"mean": 120, "median": 120},
            "contrast": {"mean": 60, "median": 65},
            "fourier_descriptors": {
                "mean": [1.0, 0.25, 0.15, 0.06, 0.03, 0.04, 0.03, 0.02, 0.01, 0.01],
                "max": [1.0, 0.6, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
                "min": [1.0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
            }
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


    
    formatted_prompt = UNIFIED_ANALYSIS_PROMPT.format(
        # Technical quality metrics
        sharpness=image_metrics["sharpness"],
        brightness=image_metrics["brightness"],
        contrast=image_metrics["contrast"],
        fourier_descriptors=str(image_metrics["fourier_descriptors"]),
        training_fourier_descriptors=str(training_image_metrics["fourier_descriptors"]),
        
        # Training data references
        training_sharpness_mean=training_image_metrics["sharpness"]["mean"],
        training_sharpness_median=training_image_metrics["sharpness"]["median"],
        training_brightness_mean=training_image_metrics["brightness"]["mean"],
        training_brightness_median=training_image_metrics["brightness"]["median"],
        training_contrast_mean=training_image_metrics["contrast"]["mean"],
        training_contrast_median=training_image_metrics["contrast"]["median"],
        
        # Prediction data
        confidence_score=confidence_score,
        robustness_status=robustness_status,
        stability_score=stability_score,
        anatom_site=anatom_site if anatom_site else "unknown",
        age_approx=age_approx if age_approx else "unknown",
        sex=sex if sex else "unknown",
        
        # Reference evaluation metrics
        tp_confidence_mean=evaluation_metrics["categories"]["TP"]["confidence_scores"]["mean"],
        fp_confidence_mean=evaluation_metrics["categories"]["FP"]["confidence_scores"]["mean"],
        tn_confidence_mean=evaluation_metrics["categories"]["TN"]["confidence_scores"]["mean"],
        fn_confidence_mean=evaluation_metrics["categories"]["FN"]["confidence_scores"]["mean"]
    )
    global latest_analysis_response
    latest_analysis_response = None
    try:
        # Run the agent with the formatted prompt
        response = agent.run(formatted_prompt)
        
        # If using the enhanced agent, get results directly
        if hasattr(agent, 'get_tool_result'):
            analysis_result = agent.get_tool_result(tool_name="analysis_response")
            if analysis_result is not None:
                return analysis_result
                
        # Best effort to extract the result from the observations
        if hasattr(agent, 'memory') and agent.memory.steps:
            latest_step = agent.memory.steps[-1]
            if hasattr(latest_step, 'observations'):
                # If observations is a string, try to extract and parse
                observations = latest_step.observations
                if observations and isinstance(observations, str):
                    import json
                    # Try using regex to find a JSON object
                    import re
                    match = re.search(r'Observations:\s*({.*})', observations, re.DOTALL)
                    if match:
                        try:
                            return json.loads(match.group(1))
                        except:
                            pass
                    
                    # Try plain eval as fallback
                    try:
                        import ast
                        return ast.literal_eval(observations)
                    except:
                        pass
                
                # If we got here, we either couldn't parse observations as a dict
                # or observations wasn't a string
                print(f"Could not parse observations as dict: {observations}")
        
        # Fallback
        return create_fallback_response()
    except Exception as e:
        print(f"❌ Error during agent analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return create_fallback_response()