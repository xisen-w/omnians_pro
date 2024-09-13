import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.basicAgents import LLMAgent
from typing import List, Dict
import json

class EssayCompiler(LLMAgent):
    def __init__(self, model_name):
        super().__init__(model_name)

    def compile_essay(self, essay_structure: List[Dict], compiled_essay: str) -> str:
        writing_style = """
        Writing Style & Tone: 
            1.	Concise and Objective Statements: The text presents research findings and observations in a clear, factual manner, without emotional language. Examples include direct statements like “Studies on protests slogans… remain scarce.”
            2.	Citation of Sources: The writing frequently refers to specific academic sources (e.g., “Nassar and Al-Harahsheh, 2020,” “Srour, 2021”), ensuring arguments are well-supported by previous literature.
            3.	Interdisciplinary Approach: The content draws from various fields, such as linguistics, sociology, musicology, and political theory, showing a comprehensive and integrated understanding of the subject matter.
            4.	Use of Specialized Terminology: Terms like “socio-pragmatic functions,” “Ideology Schema,” and “nonviolent discipline” indicate a specialized vocabulary targeted at readers familiar with academic discourse.
            5.	Explanatory Tone: There is a tendency to explain or clarify concepts, such as the differences between slogans and chants, or the significance of certain phrases used in protests.
            6.	Balanced Analysis: The text includes balanced discussions, often referencing multiple perspectives or cases (e.g., comparisons between Lebanese and Iraqi protest slogans, or the impact of nonviolent actions).

        Good Writing Samples: (Look for the way things are written | Hypothetical. These are NOT RELATED WITH THE ESSAY. ONLY LEARN FROM THE STYLE. ) 

            1.	“Studies on protests slogans in other countries, including Lebanon and Iraq, remain scarce. For example, Nassar and Al-Harahsheh (2020) explore the socio-pragmatic functions of 202 slogans raised by Lebanese protestors in 2019 and 2020.”
            2.	“Srour (2021) has researched their ideological features drawing on van Dijk’s (1997) Ideology Schema.”
            3.	“An exception is a recent work by Lovotti and Proserpio (2021), which interviews some of the activists who participated in the first wave of the protests to investigate the factors that shaped their decisions and led to different forms of spontaneous participation.”
            4.	“Slogans are shorter than chants and can be communicated through different mediums, including ‘banners, wall graffiti, audio-visual instruments, chanting, speeches and songs’ (Al-Sowaidi et al., 2017: 622).”
            5.	“The slogan was a short yet effective phrase that powerfully captured the collective grievances of all Iraqis, no matter their background or secondary identities.”
            6.	“The insult poetry was evidenced by the use of three words: qumamat [trash], sarsary [corrupt or immoral] and thail [tail]. The three slang words collectively insulted all those who have ruled Iraq post-2003.”
            7.	“These sonic events first occurred unexpectedly and ironically in the demonstration sites, but were nonetheless grasped and performed as a political act, functioning as nonviolent weapons used to ‘attack’ political opponents.”
            8.	“Drawing from insights in musicology, sound studies, and critical theory, this article adopts an interdisciplinary approach to study the role of these unexpected sounds in the Umbrella Movement.”
            9.	“Gene Sharp suggests that resisters must ‘stand together’ to maintain ‘nonviolent discipline, internal solidarity, and morale, and to continue the struggle.’”
            10.	“Rather, I wish to recover what might be valuable about the question by reorienting the question of where the protest songs are toward a broader inquiry into sound’s relationship to dissent.”

        """

        #TODO Make it dynamic. 

        """
        Compile an essay using the provided structure, paragraphs, and citations.
        Ensure smooth transitions between sections and append citations at the bottom.
        
        Args:
            essay_structure (List[Dict]): The structured sections of the essay.
            paragraphs (List[str]): The compiled paragraphs for each section.
            citations (List[str]): The list of citations for the essay.

        Returns:
            str: The final compiled essay with citations.
        """
        system_prompt = """
        You are a professional writer and essay compiler. Your job is to merge the given paragraphs into a coherent essay, 
        ensuring smooth transitions between sections. Make sure that the flow between sections is seamless and the essay 
        reads naturally. At the end of the essay, include all the references in the format provided.

        For each paragraph:
        - Keep the paragraph structure and logic intact. But if you think some paragraphs are better to be merged togethere for deeper depth. Please do.
        - Ensure transitions between sections are smooth and logical. Deliver a uniform story/theme in great depth.

        After compiling the paragraphs, append the provided citations at the end, under a section titled "References". Do not hallucinate.
        """

        # Prepare the input to the model (prompt for the essay compilation)
        input_prompt = f"""
        Here is the structure of the essay, the corresponding paragraphs, and the list of citations:

        Essay Structure:
        {json.dumps(essay_structure, indent=2)}

        Compiled Paragraphs & Citations
        {compiled_essay}

        Please compile this into a single coherent essay with smooth transitions between sections.
        Make sure to include a "References" section at the end with all the citations.

        Output the result in a proper essay format. Do not omit anything. Deliver a consistent story in-depth in a particular theme. 

        Writing Style: {writing_style}

        """#TODO Dynamic Writing Style in the future. 

        # Perform the action using the LLM model
        response = self.perform_action(
            user_prompt=input_prompt,
            system_prompt=system_prompt
        )

        return response.content if response else "Compilation failed. Please try again."

# # Example usage:
# if __name__ == "__main__":
#     # Initialize the agent
#     model_name = "gpt-4o-mini"
#     essay_compiler_agent = EssayCompiler(model_name)

#     # Sample input data (replace with actual data)
#     essay_structure = [
#         {"section": "Introduction", "purpose": "Introduce the topic and provide a thesis statement."},
#         {"section": "Body", "purpose": "Discuss the main arguments with supporting evidence."},
#         {"section": "Conclusion", "purpose": "Summarize the key points and provide final thoughts."}
#     ]
#     paragraphs = [
#         "This is the introductory paragraph explaining the purpose of the essay...",
#         "The body paragraph expands on the arguments with evidence provided...",
#         "Finally, the conclusion ties everything together and reiterates the thesis..."
#     ]
#     citations = [
#         "Reference 1: Author A. (2020). Title of the Article. Journal Name, 15(3), 200-210.",
#         "Reference 2: Author B. (2021). Another Title. Another Journal, 10(1), 100-115."
#     ]

#     # Compile the essay
#     # Convert the paragraphs list into a single string
#     compiled_essay = "\n\n".join(paragraphs) + "\n\n".join(citations)
    
#     better_essay = essay_compiler_agent.compile_essay(essay_structure, compiled_essay)

#     # Print the final essay
#     print("Compiled Essay:")
#     print(better_essay)

