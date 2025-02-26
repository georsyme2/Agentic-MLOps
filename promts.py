# promts.py - Complete replacement
import json

# Technical Image Quality Analysis Prompt
TECHNICAL_QUALITY_PROMPT = """
Analyze the technical quality of this image by comparing these metrics to the training data:
- Current image sharpness: {sharpness} (Training mean: {training_sharpness_mean}, median: {training_sharpness_median})
- Current image brightness: {brightness} (Training mean: {training_brightness_mean}, median: {training_brightness_median})
- Current image contrast: {contrast} (Training mean: {training_contrast_mean}, median: {training_contrast_median})

For each metric, determine:
1. Is it within an acceptable range compared to training data?
2. How much does it deviate from the expected values (as a percentage)?
3. What specific impact might this have on the model's accuracy?
4. What technical improvements would you recommend?

Provide a concise analysis of technical quality issues and their potential impact.
"""

# Image Content Analysis Prompt
# Image Content Analysis Prompt
IMAGE_CONTENT_PROMPT = """
Analyze if this image likely contains proper skin lesion content by comparing Fourier descriptors:

Current image descriptors: {fourier_descriptors}
Training set descriptor patterns: {training_fourier_descriptors}

Since Fourier descriptor can have a large span of values, be carefull and analyze every discriptor separately not only with the mean but onlso with min and max values.
If the image is not a skin lesion, most of the Fourier descriptors will not be in the range of Fourier descriptors of the training data. 

Your assessment should be at least 2-3 sentences explaining your reasoning about whether this appears to be a valid skin lesion image.
"""

# Prediction Reliability Analysis Prompt
PREDICTION_ANALYSIS_PROMPT = """
Analyze this prediction's reliability:
- Confidence score: {confidence_score:.4f}
- Robustness status: {robustness_status}
- Anatomical site: {anatom_site}
- Patient age: {age_approx}
- Patient sex: {sex}

Relevant comparison metrics from evaluation data:
- True Positive confidence mean: {tp_confidence_mean}
- False Positive confidence mean: {fp_confidence_mean}
- True Negative confidence mean: {tn_confidence_mean}
- False Negative confidence mean: {fn_confidence_mean}

Provide insights on:
1. How this confidence score compares to historical patterns for correct and incorrect predictions
2. Whether the robustness status indicates reliability
3. If demographic factors might affect prediction accuracy based on historical patterns

Keep your analysis concise and focused on reliability assessment.
"""

# Final Decision Synthesis Prompt
DECISION_PROMPT = """
Based on these analyses:

TECHNICAL QUALITY ANALYSIS:
{technical_quality_analysis}

CONTENT ANALYSIS:
{content_analysis}

PREDICTION RELIABILITY ANALYSIS:
{prediction_analysis}

Synthesize a final decision using ONLY the following JSON format and nothing else:
{{
    "assessment_summary": "One sentence conclusion about image quality and prediction reliability",
    "image_feedback": "Brief feedback about image quality issues",
    "content_assessment": "Brief assessment of whether the image contains valid skin lesion content",
    "prediction_feedback": "Brief feedback about prediction reliability",
    "user_suggestion": "Primary recommendation for the user",
    "override_decision": {{
        "override": true/false,
        "corrected_prediction": null or 0 or 1
    }}
}}

IMPORTANT: Response must be VALID JSON only. No additional text, explanations, or non-JSON content.
"""