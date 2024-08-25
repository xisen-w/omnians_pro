from utils.schemas import EssayStructureSchema
from Agents.basicAgents import LLMAgent
import json

class StructureOutliner(LLMAgent):
    def __init__(self, model_name):
        super().__init__(model_name)

    def structure_essay(self, main_question: str, sub_questions: list[str]):
        system_prompt = f"""
        You are professional. You are in charge of structuring an essay that answers this question:

        {main_question}

        Your colleague has brainstormed a list of sub-questions related to this essay. Take advantage of these and structure your argument. Your output should be a list of paragraphs that include:

        1) What the paragraph is for.
        2) What evidence is needed.
        3) How is the argument developed. 

        Do not generate anything else except the list.

        List of sub-questions to explore:

        {sub_questions}

        An Example Output Should be:

        [
            {{
                "section": "Introduction",
                "purpose": "To introduce the topic and provide a thesis statement.",
                "evidence_needed": "General background information on digitization and simulation, including definitions and context.",
                "argument_development": "Explain the increasing relevance of digitizing and simulating the physical world, highlighting its significance in modern technology and society."
            }},
            ... (additional sections following this format)
        ]
        """

        user_prompt = main_question

        # Use the perform_action method from LLMAgent
        response = self.perform_action(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema_class=EssayStructureSchema
        )

        return response.essay_structure if response else None

    def to_string(self):
        """
        Convert the LLM response to a readable string.
        """
        if self.previous_result:
            return json.dumps(self.previous_result.dict(), indent=2)
        return "No result available."
