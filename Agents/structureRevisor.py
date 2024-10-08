import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.schemas import EssayStructureSchema
from Agents.basicAgents import LLMAgent

class StructureRevisor(LLMAgent):
    def __init__(self, model_name):
        super().__init__(model_name)

    def revise_outline(self, question: str, provided_outline: str, context_summaries: list):
        prompt_template = """
        You are a professor from Oxford. You are in charge of revising the first draft of a structure of an essay written by a bright student. 
        The student has not read anything before, so you will pay attention to combining the context into the structure to update a better one.

        Note: try to develop sections in a way only 3-4 big ideas are talked about, and the smllar sections are just complementary components for discussing the big ideas.

        The question to answer is: 
        {question}

        A less qualified outline is listed here. It is not that good because it is often too general and there's no uniform theme. 
        {provided_outline}

        Using the summarised context here, writer a much better structure (please do cite and indicate specific literature as sources for evidence):

        {context_summaries}

        Revise and output the new structure accordingly. Make sure that nothing other than the new structure is output.
        """

        # Format the user prompt as a single string
        system_prompt = prompt_template.format(
            question=question,
            provided_outline=provided_outline,
            context_summaries="\n".join(context_summaries)
        )

        # Use the perform_action method from LLMAgent 
        response = self.perform_action(
            system_prompt=system_prompt,
            user_prompt="Stay critical. Restructure the whole essay from a new perspective, unless you think that the current one is very very good. Think step by step. Stay very critical and constructive. Your responsibility is to comprehend the background information and distill these information to generate a much better framework.",
            schema_class=EssayStructureSchema,
        )

        return response if response else None

# if __name__ == "__main__":
#     model_name = "gpt-4o-mini-2024-07-18"
#     revisor = StructureRevisor(model_name)

#     question = "What are the main challenges in climate change policy?"
#     provided_outline = """
#     1. Introduction
#         - Define climate change
#     2. Policy Overview
#         - Brief history of climate change policies
#     3. Challenges
#         - Economic challenges
#         - Political challenges
#     4. Conclusion
#     """
#     context_summaries = [
#         "The IPCC report (2023) emphasizes the economic impact of delaying climate action.",
#         "Research from Oxford (2021) highlights political resistance in several key countries."
#     ]

#     # Call the revise_outline method and print the result
#     revised_outline = revisor.revise_outline(question, provided_outline, context_summaries)

#     # Print the revised outline to test
#     print("Revised Outline:\n", revised_outline)