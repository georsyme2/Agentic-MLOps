from tensorflow.keras.models import load_model
from smolagents import ToolCallingAgent, HfApiModel, TransformersModel

from agent_utils import ImageQualityAssessmentTool, PredictionEvaluationTool

def initialize_model(model_path="best_model.h5"):
    """
    Initializes and returns the trained Keras model.
    """
    model = load_model(model_path)
    return model

def initialize_agent():  # Replace with your actual API key
    """
    Initializes and returns the AI agent.
    """
    # llm_model = HfApiModel(model_id="meta-llama/Llama-3.3-70B-Instruct", token=api_key, timeout=300)
    llm_model = TransformersModel(model_id="Qwen/Qwen2.5-7B-Instruct",device_map="cuda")
    agent = ToolCallingAgent(
        tools=[],
        model=llm_model
    )
    return agent