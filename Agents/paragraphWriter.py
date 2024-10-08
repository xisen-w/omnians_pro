import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pydantic import BaseModel, Field
from typing import List
from .basicAgents import LLMAgent
import json

# Define the schema for paragraph compilation output
class ParagraphCompilation(BaseModel):
    paragraph: str = Field(..., description="The compiled content of the paragraph.")
    references: List[str] = Field(..., description="A list of references used in the paragraph.")

# Define the schema for the entire essay structure
class EssayCompilationSchema(BaseModel):
    essay: List[ParagraphCompilation] = Field(..., description="The compiled essay with paragraphs and references.")

# Define the agent class for compiling paragraphs and the full essay
class ParagraphWriter(LLMAgent):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def compile_paragraph(self, paragraph_structure: dict, context: List[dict]) -> ParagraphCompilation:
        """
        Compile a single paragraph based on its structure and relevant context.
        
        Parameters:
        - paragraph_structure: dict containing the structure for a specific paragraph.
        - context: List of dictionaries containing retrieved context (text, source, page number).
        
        Returns:
        - ParagraphCompilation: The compiled paragraph with references.
        """
        system_prompt = """
        You are tasked with writing a paragraph as part of an essay. You are given the structure of the paragraph and relevant context from sources like PDFs. Write a professional and compelling paragraph based on the structure and cite the sources appropriately.
        """
        
        user_prompt = f"""
        Paragraph Structure:
        {paragraph_structure}
        
        Context from PDFs:
        {json.dumps(context, indent=2)}
        
        Your task is to use the structure and the context to write a detailed, high-quality paragraph, including in-text citations for evidence. Use APA style for citations with the format (Author, Year, Page Number). Also, generate a list of references based on the citations.
        """

        # Perform the action and parse the result using the ParagraphCompilation schema
        response = self.perform_action(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema_class=ParagraphCompilation
        )
        
        return response if response else None

    def compile_entire_essay(self, essay_structure: List[dict], context_list: List[List[dict]]) -> EssayCompilationSchema:
        """
        Compile the entire essay by iterating through the paragraph structures.
        
        Parameters:
        - essay_structure: List of dictionaries containing the structure for each paragraph.
        - context_list: List of lists of dictionaries containing context (text, source, page) for each paragraph.
        
        Returns:
        - EssayCompilationSchema: The compiled essay with paragraphs and references.
        """
        compiled_paragraphs = []
        
        for idx, para_struct in enumerate(essay_structure):
            if idx < len(context_list):
                context = context_list[idx]  # Get the corresponding context for the paragraph
            else:
                context = "No specific context available for this paragraph."
            compiled_para = self.compile_paragraph(para_struct, context)
            if compiled_para:
                compiled_paragraphs.append(compiled_para)
        
        # Return the final compiled essay in the schema
        return EssayCompilationSchema(essay=compiled_paragraphs)

# Example usage
if __name__ == "__main__":
    # Mock essay structure
    essay_structure = [
        {
            "section": "Introduction",
            "purpose": "Introduce the topic of digitization.",
            "evidence_needed": "General statistics about digital transformation.",
            "argument_development": "Explain how digitization has impacted various industries."
        },
        {
            "section": "Impact on Education",
            "purpose": "Discuss digitization's role in modern education.",
            "evidence_needed": "Studies on digital learning tools, online education, and student outcomes.",
            "argument_development": "Highlight the positive changes digital tools have brought to education."
        }
    ]
    
    # Mock context for each paragraph (usually retrieved from PDFs or other sources)
    context_list = [
        [
            {"text": "Digitization has transformed industries globally.", "source": "Industry Report", "page": 5},
            {"text": "Digital tools have been widely adopted in education.", "source": "Education Study", "page": 12}
        ],
        [
            {"text": "Online education has increased access to learning.", "source": "Learning Revolution", "page": 22},
            {"text": "Student outcomes have improved due to digital learning.", "source": "Education Research", "page": 7}
        ]
    ]

    # Initialize the agent
    compilor = ParagraphWriter(model_name="gpt-4o-mini")

    # Compile the entire essay
    compiled_essay = compilor.compile_entire_essay(essay_structure, context_list)
    
    # Output the result
    print(json.dumps(compiled_essay.dict(), indent=2))