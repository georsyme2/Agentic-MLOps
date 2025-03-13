# agent_tools.py
import json
from smolagents import Tool

# Global variable to store the latest response
latest_analysis_response = None

class AnalysisResponseTool(Tool):
    name = "analysis_response"
    description = "Provides a detailed analysis of image quality, content, and prediction reliability"
    inputs = {
        "assessment_summary": {"type": "string", "description": "One sentence conclusion about image quality and prediction reliability"},
        "image_quality": {"type": "string", "description": "Brief feedback about image quality issues"},
        "content_assessment": {"type": "string", "description": "Brief assessment of whether the image contains valid skin lesion content"},
        "prediction_reliability": {"type": "string", "description": "Brief feedback about prediction reliability"},
        "user_suggestion": {"type": "string", "description": "Primary recommendation for the user"},
        "override_recommended": {"type": "boolean", "description": "Whether to override the prediction"},
        "corrected_prediction": {"type": "integer", "description": "Corrected prediction value (0, 1, or null) if override is recommended", "nullable": True},
        "technical_quality_details": {"type": "string", "description": "Detailed analysis of image quality issues", "nullable": True},
        "content_details": {"type": "string", "description": "Detailed analysis of image content", "nullable": True},
        "reliability_details": {"type": "string", "description": "Detailed analysis of prediction reliability", "nullable": True}
            }
    output_type = "object"

    def forward(self, assessment_summary, image_quality, content_assessment, 
                prediction_reliability, user_suggestion, override_recommended, 
                corrected_prediction=None, technical_quality_details="", 
                content_details="", reliability_details=""):
        """Returns a structured analysis response"""
        # Convert corrected_prediction to proper type
        if override_recommended and corrected_prediction is not None:
            corrected_prediction = int(corrected_prediction)
        else:
            corrected_prediction = None
            
        result = {
            "assessment_summary": assessment_summary,
            "image_feedback": image_quality,
            "content_assessment": content_assessment,
            "prediction_feedback": prediction_reliability,
            "user_suggestion": user_suggestion,
            "override_decision": {
                "override": override_recommended,
                "corrected_prediction": corrected_prediction
            },
            "detailed_analyses": {
                "technical_quality": technical_quality_details,
                "content": content_details,
                "reliability": reliability_details
            }
        }




        return result