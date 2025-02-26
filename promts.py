import json
IMAGE_QUALITY_PROMPT = """
    The following image quality metrics have been computed for the new image:
    {image_metrics}

    The training set contains the following image quality statistics:
    {training_image_metrics}

    The evaluation metrics log from previous predictions contains:
    {evaluation_metrics}

    Thoroughly analyze the image by comparing its metrics to the training data:
    
    1. TECHNICAL QUALITY: Compare sharpness, brightness, contrast with training set ranges. 
       - For each metric that deviates, quantify HOW MUCH it deviates (in percentages compared to mean/median)
       - Determine the IMPACT of each deviation on prediction accuracy based on evaluation metrics
       - Suggest SPECIFIC technical improvements for each issue (e.g., lighting conditions, focus adjustments)
    
    2. CONTENT ANALYSIS: Analyze the Fourier descriptors to determine if the image contains features expected in skin lesions.
       - Identify if the descriptor patterns match typical skin lesion patterns
       - Compare to patterns seen in successful vs. unsuccessful predictions from evaluation data
       - Determine if the image likely contains a skin lesion or potentially different content
    
    Respond ONLY in JSON format:
    {{
        "technical_quality": {{
            "overall_assessment": "Good/Acceptable/Poor",
            "sharpness": {{
                "status": "Within range/Too low/Too high",
                "deviation": "Percentage from mean/median",
                "impact": "How this affects prediction accuracy based on evaluation data",
                "improvement": "Specific camera or positioning advice"
            }},
            "brightness": {{
                "status": "Within range/Too dark/Too bright",
                "deviation": "Percentage from mean/median",
                "impact": "How this affects prediction accuracy based on evaluation data",
                "improvement": "Specific lighting advice"
            }},
            "contrast": {{
                "status": "Within range/Too low/Too high",
                "deviation": "Percentage from mean/median",
                "impact": "How this affects prediction accuracy based on evaluation data",
                "improvement": "Specific contrast adjustment advice"
            }}
        }},
        "content_assessment": {{
            "contains_skin_lesion": "Likely/Uncertain/Unlikely",
            "descriptor_match": "How well the descriptors match training data",
            "typical_performance": "How the model performs on images with similar descriptors",
            "explanation": "Analysis of key Fourier patterns that indicate skin lesion presence/absence"
        }},
        "priority_issues": ["Ranked list of most critical issues to address"]
    }}
    """

PREDICTION_PROMPT = """
    The model made a prediction with a confidence score of {confidence_score:.4f}.
    The robustness check determined that the model's prediction was "{robustness_status}".
    
    Metadata for this image:
    - Sex: {sex}
    - Age: {age_approx}
    - Anatomical Site: {anatom_site}

    The evaluation metrics log from previous predictions contains:
    {evaluation_metrics}

    Conduct a comprehensive analysis of this prediction's reliability:
    
    1. CONFIDENCE ANALYSIS:
       - Compare current confidence ({confidence_score:.4f}) to historical mean/distribution for both correct and incorrect predictions
       - Determine if this confidence level typically indicates a reliable prediction
    
    2. ROBUSTNESS EVALUATION:
       - Analyze the "{robustness_status}" determination - a stable prediction that doesn't change with minor image alterations indicates reliability
       - Understand that an "Unstable" status is a significant concern suggesting the model is not confident in its classification
    
    3. METADATA COMPATIBILITY:
       - Compare with historical performance for this specific anatomical site
       - Analyze if age and sex demographics influence prediction reliability in similar cases
       - Identify if this demographic combination shows any patterns in the evaluation metrics
    
    Respond ONLY in JSON format:
    {{
        "confidence_analysis": {{
            "relative_position": "Where this score falls in historical distribution",
            "typical_accuracy": "Historical accuracy rate for predictions with similar confidence",
            "historical_comparison": "How this compares to TP/FP/TN/FN confidence patterns"
        }},
        "robustness_assessment": {{
            "status_evaluation": "Whether the prediction remains stable with minor image variations",
            "reliability_impact": "How the stability/instability affects trust in the prediction"
        }},
        "metadata_analysis": {{
            "anatomical_site_performance": "Historical accuracy for this anatomical site",
            "demographic_factors": "How age and sex might influence prediction accuracy",
            "comparison": "How this case compares to similar demographic cases"
        }},
        "reliability_score": {{
            "trust_level": "High/Medium/Low",
            "explanation": "Detailed rationale based on confidence, robustness, and metadata"
        }}
    }}
    """

DECISION_PROMPT = """
    Given the following information:
    - Image Quality Assessment: {image_quality_response}
    - Prediction Confidence Comparison: {prediction_response}

    Synthesize a comprehensive final analysis that clearly separates and addresses:
    
    1. IMAGE ISSUES vs. PREDICTION RELIABILITY
       - Distinguish between problems in the input data versus problems in the model's processing
       - Determine if poor quality is the main issue or if the model is struggling with this type of classification
    
    2. ACTIONABLE RECOMMENDATIONS
       - Provide concrete, step-by-step instructions for improvement
       - Prioritize recommendations based on impact (what will most improve diagnostic accuracy)
    
    3. OVERRIDE DECISION
       - Determine if the prediction should be trusted based on image quality and confidence analysis
       - Consider overriding if image quality is good but confidence is unusually low/high compared to historical patterns
       - Consider overriding if robustness check shows instability

    Respond ONLY in JSON format:
    {{
        "assessment_summary": "Brief overall conclusion about image quality and prediction reliability",
        
        "image_quality": {{
            "overall_rating": "Optimal/Acceptable/Problematic/Unsuitable",
            "key_issues": ["Prioritized list of specific image quality problems"],
            "impact_on_reliability": "How the image quality affects prediction reliability"
        }},
        
        "prediction_reliability": {{
            "trust_level": "High/Medium/Low",
            "confidence_assessment": "Analysis of the prediction confidence compared to historical patterns",
            "robustness_factor": "How prediction stability affects reliability"
        }},
        
        "recommendations": {{
            "priority_actions": ["Numbered step-by-step actions for the user"],
            "technical_improvements": ["Specific camera, lighting, positioning suggestions"],
            "acquisition_guidance": "Advice for better image acquisition"
        }},
        
        "override_decision": {{
            "override": true/false,
            "reasoning": "Explanation for override decision",
            "corrected_prediction": "0 or 1 (if applicable)"
        }}
    }}
    """