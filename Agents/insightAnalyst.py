from utils.schemas import SubQuestionSchema
from Agents.basicAgents import LLMAgent

class InsightAnalyst(LLMAgent):
    def __init__(self, model_name):
        super().__init__(model_name="gpt-4o")  # Initialize the base LLMAgent class

    def generate_sub_questions(self, research_question: str):
        prompt_template = """
        You are an expert in brainstorming, analyzing, and researching. You are given a research question: 

        {user_prompt}

        Your task is to generate a list of sub-questions that are related to this and will help in building a professional research thesis.

        You are here to do deeep research. Hence, the questions should be such that they are not obvious and require a lot of research to answer. 
        
        They should target the research question in different aspects, yet gradually increasing the depth and complexity of the questions to ask for details.
        """

        # Use the perform_action method from LLMAgent
        response = self.perform_action(
            system_prompt=prompt_template,
            schema_class=SubQuestionSchema,
            user_prompt=research_question
        )

        return response.sub_questions if response else None
    
    
