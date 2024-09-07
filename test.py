# Import necessary modules (Assuming openai and LLMResponsePro are defined in your environment)
from openai import OpenAI
from fundations.LLMResponsePro import LLMResponsePro
from pydantic import BaseModel, Field

# Initialize the LLMResponsePro instance with the desired model name
model_name = "gpt-4o-mini-2024-07-18"  # Replace with your desired model name
llm_response_pro = LLMResponsePro(model_name)

# Define a Pydantic schema with descriptive fields
class LearningSchema(BaseModel):
    supervised_learning: str = Field(..., description="Explanation of supervised learning")
    unsupervised_learning: str = Field(..., description="Explanation of unsupervised learning")
    key_differences: str = Field(..., description="Key differences between supervised and unsupervised learning")

# Define your prompts
user_prompt = "What are the key differences between supervised and unsupervised learning?"
system_prompt = "Provide a brief and clear explanation suitable for a technical audience."

# Get the structured output
response = llm_response_pro.structured_output(
    schema_class=LearningSchema,
    user_prompt=user_prompt,
    system_prompt=system_prompt
)

# Print the response
print(response)
