import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.basicAgents import LLMAgent
from typing import List, Dict
import json

class CritiqueAgent(LLMAgent):
    def __init__(self, model_name):
        super().__init__(model_name)

    def critique_essay(self, essay_structure: List[Dict], compiled_essay: str) -> str:
        """
        Provide a detailed critique of the essay structure and content.

        Args:
            essay_structure (List[Dict]): The structured sections of the essay.
            compiled_essay (str): The entire essay text that needs to be critiqued.

        Returns:
            str: A detailed critique highlighting strengths, weaknesses, and suggestions for improvement.
        """

        critique_criteria = """
        Criteria for Critique:
            1. Structure & Flow: Is the essay well-organized? Are the sections clearly delineated? Are the transitions between sections smooth and logical?
            2. Clarity: Is the essay easy to follow? Are complex ideas explained clearly?
            3. Argument Depth: Are the arguments well-developed? Is there enough depth in the reasoning provided?
            4. Use of Evidence: Does the essay provide sufficient evidence to support its claims? Are the sources cited properly?
            5. Writing Quality: Is the essay well-written? Are there any awkward sentences or grammatical issues?
            6. Citation & Referencing: Are all citations properly formatted and placed? Are references provided at the end, and do they follow academic standards?

        Example Critique Structure:
            - Strengths: Highlight what the essay does well in terms of argument structure, clarity, use of evidence, etc.
            - Weaknesses: Point out areas where the essay falls short, such as unclear reasoning, weak transitions, or insufficient evidence.
            - Suggestions: Offer constructive feedback on how the essay can be improved, such as providing more detailed evidence, improving transitions, or fixing citation issues.
        """

        system_prompt = """
        You are a professional essay reviewer. Your task is to carefully critique the provided essay based on its structure, clarity, argument depth, use of evidence, writing quality, and citation style.
        
        Follow this format:
            1. Identify strengths and weaknesses in the essay.
            2. Suggest ways to improve its structure, arguments, and coherence.
            3. Offer guidance on improving the writing quality and citations.

        Make sure your critique is detailed, constructive, and professional.

        Your critique should address the following key areas:
        - Structure & Flow
        - Clarity
        - Argument Depth
        - Use of Evidence
        - Writing Quality
        - Citation & Referencing
        """

        # Prepare the input to the model (prompt for the critique)
        input_prompt = f"""
        Here is the structure of the essay and the compiled essay text that needs to be critiqued:

        Essay Structure:
        {json.dumps(essay_structure, indent=2)}

        Compiled Essay:
        {compiled_essay}

        Please provide a detailed critique following the critique criteria. Focus on areas such as structure, clarity, argument depth, use of evidence, and citation style.

        Writing Style: {critique_criteria}
        """

        # Perform the action using the LLM model
        response = self.perform_action(
            user_prompt=input_prompt,
            system_prompt=system_prompt
        )

        return response.content if response else "Critique failed. Please try again."


# Example usage:
if __name__ == "__main__":
    # Initialize the agent
    model_name = "gpt-4o-mini"
    critique_agent = CritiqueAgent(model_name)

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

    # Critique the essay
    critique = critique_agent.critique_essay(essay_structure, compiled_essay)

    # Print the critique
    print("Essay Critique:")
    print(critique)