# initialization.py
from tensorflow.keras.models import load_model
from smolagents import ToolCallingAgent, TransformersModel
from agent_tools import AnalysisResponseTool
from custom_agent import EnhancedToolCallingAgent

def initialize_model(model_path="best_model.h5"):
    """
    Initializes and returns the trained Keras model.
    """
    model = load_model(model_path)
    return model

def initialize_agent():
    """
    Initializes and returns the AI agent with analysis tools.
    """
    # Use TransformersModel for local inference
    llm_model = TransformersModel(model_id="Qwen/Qwen2.5-7B-Instruct", device_map="cuda")
    
    # Create the agent with the analysis tool
    agent = EnhancedToolCallingAgent(
        tools=[AnalysisResponseTool()],
        model=llm_model
    )
    return agent