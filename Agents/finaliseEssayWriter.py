import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.basicAgents import LLMAgent
from typing import List, Dict
import json

class FinaliseEssayWriter(LLMAgent):
    def __init__(self, model_name):
        super().__init__(model_name)

    def finalise_essay(self, essay_structure: List[Dict], compiled_essay: str, critique: str) -> str:
        """
        Revise and finalise the essay based on the critique provided.

        Args:
            essay_structure (List[Dict]): The structured sections of the essay.
            compiled_essay (str): The essay that needs to be revised.
            critique (str): The critique or feedback to guide the revision process.

        Returns:
            str: The final revised version of the essay, addressing all feedback.
        """

        finalisation_criteria = """
        Finalisation Criteria:
            1. Address all weaknesses mentioned in the critique, including structure, clarity, and argument depth.
            2. Implement suggestions for improving transitions, enhancing evidence, and correcting any citation issues.
            3. Ensure smooth flow and coherence across the entire essay.
            4. Double-check that all citations are in the correct format and properly referenced in the final "References" section.
        """

        system_prompt = """
        You are a professional essay finaliser. Your task is to revise the provided essay based on the critique 
        and ensure that all feedback is addressed. Make sure the final essay is polished, coherent, and well-written, 
        with appropriate citations.

        Your finalisation should:
            - Address all issues raised in the critique, especially regarding structure, clarity, argument depth, and use of evidence.
            - Ensure smooth transitions between sections and paragraphs, making the essay read naturally.
            - Ensure all citations are correctly formatted and included in a "References" section at the end. Reduce Repetitions. 
        """

        # Prepare the input to the model (prompt for finalising the essay)
        input_prompt = f"""
        Here is the structure of the essay, the compiled essay, and the critique that needs to be addressed:

        Essay Structure:
        {json.dumps(essay_structure, indent=2)}

        Compiled Essay:
        {compiled_essay}

        Critique:
        {critique}

        Please revise and finalise the essay based on the provided critique, ensuring all feedback is addressed.
        Make sure to include a "References" section at the end with all citations.

        Writing Style: {finalisation_criteria}
        """

        # Perform the action using the LLM model
        response = self.perform_action(
            user_prompt=input_prompt,
            system_prompt=system_prompt
        )

        return response.content if response else "Finalisation failed. Please try again."


# Example usage:
if __name__ == "__main__":
    # Initialize the agent
    model_name = "gpt-4o-mini"
    finalise_essay_writer_agent = FinaliseEssayWriter(model_name)

    # Sample input data (replace with actual data)
    essay_structure = [
        {"section": "Introduction", "purpose": "Introduce the topic and provide a thesis statement."},
        {"section": "Body", "purpose": "Discuss the main arguments with supporting evidence."},
        {"section": "Conclusion", "purpose": "Summarize the key points and provide final thoughts."}
    ]
    compiled_essay = """
    This is the introductory paragraph explaining the purpose of the essay...
    The body paragraph expands on the arguments with evidence provided...
    Finally, the conclusion ties everything together and reiterates the thesis...
    """
    critique = """
    The essay is well-organized, but the transition between the introduction and the body could be smoother. 
    The arguments in the body lack sufficient depth, especially when discussing the main evidence. 
    More detailed citations are needed to strengthen the claims. The conclusion is strong, but the thesis statement 
    could be more clearly reiterated at the end.
    """

    # Finalise the essay based on the critique
    finalised_essay = finalise_essay_writer_agent.finalise_essay(essay_structure, compiled_essay, critique)

    # Print the finalised essay
    print("Finalised Essay:")
    print(finalised_essay)