from smolagents import ToolCallingAgent, ActionStep
from smolagents.memory import ToolCall  # Add this import
from typing import Any, Dict, List, Optional, Union

class EnhancedToolCallingAgent(ToolCallingAgent):
    """
    Enhanced version of ToolCallingAgent that properly captures tool outputs
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_results = {}
        
    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Override the step method to properly capture tool outputs
        """
        memory_messages = self.write_memory_to_messages()
        memory_step.model_input_messages = memory_messages.copy()
        
        try:
            # Get model response
            model_message = self.model(
                memory_messages,
                tools_to_call_from=list(self.tools.values()),
                stop_sequences=["Observation:"],
            )
            memory_step.model_output_message = model_message
            
            # Check if model called any tools
            if model_message.tool_calls is None or len(model_message.tool_calls) == 0:
                raise Exception("Model did not call any tools. Call `final_answer` tool to return a final answer.")
            
            # Extract tool call information
            tool_call = model_message.tool_calls[0]
            tool_name, tool_call_id = tool_call.function.name, tool_call.id
            tool_arguments = tool_call.function.arguments
            
        except Exception as e:
            self.logger.log(f"Error in generating tool call with model: {e}", level=0)
            raise
            
        # Create proper ToolCall objects, not just dictionaries
        memory_step.tool_calls = [ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)]
        
        # Handle final answer
        if tool_name == "final_answer":
            if isinstance(tool_arguments, dict):
                answer = tool_arguments.get("answer", tool_arguments)
            else:
                answer = tool_arguments
                
            if isinstance(answer, str) and answer in self.state.keys():
                final_answer = self.state[answer]
            else:
                final_answer = answer
                
            memory_step.action_output = final_answer
            return final_answer
            
        # Execute tool and capture output
        if tool_arguments is None:
            tool_arguments = {}
            
        # Execute the tool and store its result
        observation = self.execute_tool_call(tool_name, tool_arguments)
        
        # Store tool result in our tracking dictionary
        self.tool_results[self.step_number] = {
            "tool_name": tool_name,
            "arguments": tool_arguments,
            "result": observation
        }
        
        # Special handling for different output types
        if hasattr(observation, "__dict__"):  # Handle object outputs
            self.state[f"{tool_name}_result"] = observation
            updated_information = f"Stored result from {tool_name} in memory."
        else:
            updated_information = str(observation).strip()
            
        # Record the observation
        self.logger.log(f"Observations: {updated_information}", level=1)
        memory_step.observations = updated_information
        
        # For convenience, always store the latest result
        self.state["latest_result"] = observation
        
        return None  # This step isn't final
        
    def get_tool_result(self, step=None, tool_name=None):
        """
        Get tool result by step number or tool name
        """
        # Get the latest result if no step specified
        if step is None:
            matching_steps = list(self.tool_results.keys())
            if not matching_steps:
                return None
            step = max(matching_steps)
            
        # If we're looking for a specific tool
        if tool_name and step in self.tool_results:
            if self.tool_results[step]["tool_name"] == tool_name:
                return self.tool_results[step]["result"]
            return None
            
        # Just return result for the step
        return self.tool_results.get(step, {}).get("result")