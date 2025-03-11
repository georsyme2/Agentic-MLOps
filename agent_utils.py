import json
import os
import time
from smolagents import ToolCallingAgent, HfApiModel, Tool
from promts import DECISION_PROMPT, IMAGE_CONTENT_PROMPT, PREDICTION_ANALYSIS_PROMPT, TECHNICAL_QUALITY_PROMPT
import torch
import re
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
            
            # If we expect JSON, try to extract just the text content
            if "JSON format" in prompt and isinstance(response, str) and response.startswith('{"name":'):
                try:
                    # Extract just the text content from the JSON wrapper
                    import json
                    import re
                    
                    # Clean the response of control characters
                    cleaned_response = re.sub(r'[\x00-\x1F\x7F]', '', response)
                    parsed = json.loads(cleaned_response)
                    
                    if "arguments" in parsed and "answer" in parsed["arguments"]:
                        return parsed["arguments"]["answer"]
                    return response
                except Exception as e:
                    print(f"⚠️ Could not extract text from JSON response: {e}")
            
            # For regular text responses or if JSON extraction failed
            return response
                
        except Exception as e:
            print(f"⚠️ API call failed on attempt {attempt + 1}: {e}. Retrying in {delay} seconds...")
        
        time.sleep(delay)  # Wait before retrying

    print("❌ All retry attempts failed. Returning fallback response.")
    
    # Provide fallback responses
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

    # Step 1: Analyze technical image quality - Use direct model query
    technical_quality_prompt = TECHNICAL_QUALITY_PROMPT.format(
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
    technical_quality_analysis = direct_model_query(agent.model, technical_quality_prompt)
    
    # Step 2: Analyze image content
    content_prompt = IMAGE_CONTENT_PROMPT.format(
        fourier_descriptors=str(image_metrics["fourier_descriptors"]),
        training_fourier_descriptors=str(training_image_metrics["fourier_descriptors"])
    )
    content_analysis = direct_model_query(agent.model, content_prompt)
    
    # Step 3: Analyze prediction reliability
    prediction_prompt = PREDICTION_ANALYSIS_PROMPT.format(
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
    prediction_analysis = direct_model_query(agent.model, prediction_prompt)
    
    # Step 4: For the final decision, let's use a text-based approach instead of JSON
    final_prompt = """
    Based on these analyses:
    
    TECHNICAL QUALITY ANALYSIS:
    {technical_quality_analysis}
    
    CONTENT ANALYSIS:
    {content_analysis}
    
    PREDICTION RELIABILITY ANALYSIS:
    {prediction_analysis}
    
    Please provide your assessment in the following format:
    
    ASSESSMENT SUMMARY: One sentence conclusion about image quality and prediction reliability
    IMAGE FEEDBACK: Brief feedback about image quality issues
    CONTENT ASSESSMENT: Brief assessment of whether the image contains valid skin lesion content
    PREDICTION FEEDBACK: Brief feedback about prediction reliability
    USER SUGGESTION: Primary recommendation for the user
    OVERRIDE DECISION: yes or no
    CORRECTED PREDICTION: null, 0, or 1
    """.format(
        technical_quality_analysis=technical_quality_analysis,
        content_analysis=content_analysis,
        prediction_analysis=prediction_analysis
    )
    
    final_response = direct_model_query(agent.model, final_prompt, max_tokens=1000)
    
    # Parse the response into a structured format
    final_decision = parse_text_response(final_response)
    
    # Add detailed analyses
    final_decision["detailed_analyses"] = {
        "technical_quality": technical_quality_analysis,
        "content": content_analysis,
        "reliability": prediction_analysis
    }
    
    return final_decision

def parse_text_response(text):
    """
    Parse a text response with labeled sections into our expected structure
    """
    result = {
        "assessment_summary": "Analysis completed.",
        "image_feedback": "Image quality analysis completed.",
        "content_assessment": "Content analysis completed.",
        "prediction_feedback": "Prediction reliability assessment completed.",
        "user_suggestion": "Consider consulting a specialist.",
        "override_decision": {
            "override": False,
            "corrected_prediction": None
        }
    }
    
    # Extract information from the response
    lines = text.strip().split('\n')
    
    current_section = None
    current_text = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line starts a new section
        if ':' in line:
            section_name, content = line.split(':', 1)
            section_name = section_name.strip().upper()
            
            # Save previous section if exists
            if current_section and current_text:
                section_value = ' '.join(current_text).strip()
                if current_section == "ASSESSMENT SUMMARY":
                    result["assessment_summary"] = section_value
                elif current_section == "IMAGE FEEDBACK":
                    result["image_feedback"] = section_value
                elif current_section == "CONTENT ASSESSMENT":
                    result["content_assessment"] = section_value
                elif current_section == "PREDICTION FEEDBACK":
                    result["prediction_feedback"] = section_value
                elif current_section == "USER SUGGESTION":
                    result["user_suggestion"] = section_value
                elif current_section == "OVERRIDE DECISION":
                    result["override_decision"]["override"] = "yes" in section_value.lower()
                elif current_section == "CORRECTED PREDICTION":
                    try:
                        if "null" in section_value.lower() or "none" in section_value.lower():
                            result["override_decision"]["corrected_prediction"] = None
                        else:
                            # Try to convert to integer
                            result["override_decision"]["corrected_prediction"] = int(section_value.strip())
                    except ValueError:
                        # If we can't convert to int, check for keywords
                        if "1" in section_value or "malignant" in section_value.lower():
                            result["override_decision"]["corrected_prediction"] = 1
                        elif "0" in section_value or "benign" in section_value.lower():
                            result["override_decision"]["corrected_prediction"] = 0
            
            # Start new section
            current_section = section_name
            current_text = [content.strip()]
        elif current_section:
            # Continue previous section
            current_text.append(line)
    
    # Don't forget to process the last section
    if current_section and current_text:
        section_value = ' '.join(current_text).strip()
        if current_section == "ASSESSMENT SUMMARY":
            result["assessment_summary"] = section_value
        elif current_section == "IMAGE FEEDBACK":
            result["image_feedback"] = section_value
        elif current_section == "CONTENT ASSESSMENT":
            result["content_assessment"] = section_value
        elif current_section == "PREDICTION FEEDBACK":
            result["prediction_feedback"] = section_value
        elif current_section == "USER SUGGESTION":
            result["user_suggestion"] = section_value
        elif current_section == "OVERRIDE DECISION":
            result["override_decision"]["override"] = "yes" in section_value.lower()
        elif current_section == "CORRECTED PREDICTION":
            try:
                if "null" in section_value.lower() or "none" in section_value.lower():
                    result["override_decision"]["corrected_prediction"] = None
                else:
                    # Try to convert to integer
                    result["override_decision"]["corrected_prediction"] = int(section_value.strip())
            except ValueError:
                # If we can't convert to int, check for keywords
                if "1" in section_value or "malignant" in section_value.lower():
                    result["override_decision"]["corrected_prediction"] = 1
                elif "0" in section_value or "benign" in section_value.lower():
                    result["override_decision"]["corrected_prediction"] = 0
    
    return result

def direct_model_query(model, prompt, max_tokens=500):
    """
    Directly query the model for text generation, bypassing smolagents' JSON handling.
    
    Args:
        model: The model instance
        prompt: The text prompt
        max_tokens: Maximum tokens to generate
        
    Returns:
        str: The generated text response
    """
    try:
        # Try to access the underlying model directly
        if hasattr(model, 'model'):
            # Get the tokenizer and model
            tokenizer = model.tokenizer
            base_model = model.model
            
            # Tokenize the input with attention_mask explicitly included
            encoded_input = tokenizer(prompt, return_tensors="pt", padding=True)
            
            # Move to the appropriate device
            input_ids = encoded_input["input_ids"].to(base_model.device)
            attention_mask = encoded_input["attention_mask"].to(base_model.device)
            
            # Generate text directly with attention_mask
            with torch.no_grad():
                output = base_model.generate(
                    input_ids,
                    attention_mask=attention_mask,  # Explicitly pass attention mask
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7
                )
            
            # Decode the output
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Strip the prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
    except Exception as e:
        print(f"Direct model query failed: {e}")
    
    # Fallback to using the model's run method if direct access fails
    try:
        response = model.run(prompt)
        
        # If the response is wrapped in a JSON structure, try to extract just the text
        if isinstance(response, str) and response.startswith('{"name":'):
            try:
                import re
                import json
                
                # Clean the response of control characters
                cleaned_response = re.sub(r'[\x00-\x1F\x7F]', '', response)
                parsed = json.loads(cleaned_response)
                
                if "arguments" in parsed and "answer" in parsed["arguments"]:
                    return parsed["arguments"]["answer"]
            except:
                pass
        
        return response
    except Exception as e:
        print(f"Model run failed: {e}")
        return f"Error: Could not generate response. {str(e)}"