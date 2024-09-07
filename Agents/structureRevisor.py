from utils.schemas import EssayStructureSchema
from Agents.basicAgents import LLMAgent

class StructureRevisor(LLMAgent):
    def __init__(self, model_name):
        super().__init__(model_name)  # Initialize the base LLMAgent class

    def revise_outline(self, question: str, provided_outline: str, context_summaries: list):
        prompt_template = """
        You are a professor from Oxford. You are in charge of revising the first draft of a structure of an essay written by a bright student. 
        The student has not read anything before, so you will pay attention to combining the context into the structure to update a better one.

        Make sure that nothing other than the new structure is output.

        The question to answer is: 
        {question}

        Your student has provided you with the following outlining structure: 
        {provided_outline}

        The summarised context you will use to revise & upgrade the structure:

        {context_summaries}

        Revise and output the new structure accordingly.
        """

        # Use the perform_action method from LLMAgent
        response = self.perform_action(
            system_prompt=prompt_template,
            schema_class=EssayStructureSchema,
            user_prompt={
                "question": question,
                "provided_outline": provided_outline,
                "context_summaries": context_summaries
            }
        )

        return response.new_outline if response else None