import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.schemas import LiteratureRelationshipSchema
from Agents.basicAgents import LLMAgent

class ContextAnalyst(LLMAgent):
    def __init__(self, model_name):
        super().__init__(model_name)  # Initialize the base LLMAgent class

    def analyze_literature_structured(self, background_info: str, literature_list: list[str]):
        prompt_template = f"""
        You are an expert in literature analysis and synthesis. You are given several pieces of literature with background information:

        Background Information:
        {background_info}

        Your task is to analyze the relationship between each piece of literature and provide a detailed overview of how they support, reject, build upon, or investigate one another. Organize the analysis into clear themes and summarize the main points, identifying any contradictions, agreements, or novel contributions.

        Literature:
        {literature_list}

        Provide the analysis in clear thematic sections.
        """

        # Use the perform_action method from LLMAgent
        response = self.perform_action(
            system_prompt=prompt_template,
            user_prompt="",
            schema_class=LiteratureRelationshipSchema
        )

        return response.relationships if response else None
    
    def analyze_literature_essay(self, background_info: str, literature_list: list[str]):
        prompt_template = f"""
        You are an expert in literature analysis and synthesis from Oxford. You are given several pieces of literature with background information:

        Background Information:
        {background_info}

        Your task is to analyze the relationship between each piece of literature and provide a detailed overview of how they support, reject, build upon, or investigate one another. Organize the analysis into clear themes and summarize the main points, identifying any contradictions, agreements, or novel contributions.

        Literature:
        {literature_list}

        Provide the analysis in clear thematic sections.
        """

        # Use the perform_action method from LLMAgent
        response = self.perform_action(
            system_prompt=prompt_template,
            user_prompt="Make the essay comprehensive and effective please.",
        )

        return response.content

# Below is for individual testing. 
# background_info = "The focus is on AI-driven agent systems and their use in autonomous research models."
# literature_list = [
#     "Paper A discusses the architecture of AI agents in autonomous systems.",
#     "Paper B critiques the scalability of autonomous research models.",
#     "Paper C builds on Paper A by introducing novel learning mechanisms."
# ]

# agent = ContextAnalyst(model_name="gpt-4o-mini")
# result = agent.analyze_literature_essay(background_info, literature_list)
# print(result)