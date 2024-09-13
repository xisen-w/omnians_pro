import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.insightAnalyst import InsightAnalyst 
from Agents.structureOutliner import StructureOutliner as SR
from Agents.structureRevisor import StructureRevisor
from Agents.PDFSummaryAgent import PDFSummaryAgent
from Agents.contextAnalyst import ContextAnalyst
from Agents.paragraphWriter import ParagraphWriter, ParagraphCompilation, EssayCompilationSchema
from Agents.essayCompilor import EssayCompiler
from fundations.open_ai_RAG import Citation_Retriever

import os
import json
import hashlib
from functools import wraps
from pydantic import BaseModel

# Add these imports at the top of your file
from utils.schemas import ParagraphSchema

# Define a cache directory
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class PydanticEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.dict()
        return super().default(obj)

def cache_result(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key based only on the function name and non-object arguments
        cache_key = hashlib.md5(
            f"{func.__name__}:{str([arg for arg in args if not hasattr(arg, '__dict__')])}:{str(kwargs)}".encode()
        ).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

        # Check if cached result exists
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                print(f"Using cached result for {func.__name__}")
                cached_data = json.load(f)
                # Reconstruct Pydantic models if necessary
                if isinstance(cached_data, list) and all(isinstance(item, dict) for item in cached_data):
                    return [ParagraphSchema(**item) for item in cached_data]
                return cached_data

        # If not cached, call the function
        result = func(*args, **kwargs)

        # Cache the result
        with open(cache_file, 'w') as f:
            json.dump(result, f, cls=PydanticEncoder)

        print(f"Caching new result for {func.__name__}")
        return result
    return wrapper

# Now you can use this decorator on any function you want to cache
@cache_result
def generate_sub_questions(insight_analyst, research_question):
    return insight_analyst.generate_sub_questions(research_question)

@cache_result
def structure_essay(outliner, research_question, sub_questions):
    return outliner.structure_essay(research_question, sub_questions)

@cache_result
def summarize_pdf(pdf_summary_agent, pdf_path):
    return pdf_summary_agent.summarize_pdf(pdf_path=pdf_path)

@cache_result
def analyze_literature(context_analyst, background_info, literature_list):
    return context_analyst.analyze_literature_essay(background_info, literature_list)

@cache_result
def revise_outline(revisor, research_question, essay_structure, additional_info):
    return revisor.revise_outline(research_question, essay_structure, additional_info)

@cache_result
def retrieve_context(citation_retriever, search_key):
    answer, context, top_texts = citation_retriever.retrieve_and_ask(search_key)
    return {"answer": answer, "context": context, "top_texts": top_texts}

@cache_result
def compile_essay(paragraph_writer, essay_structure, context_list):
    return paragraph_writer.compile_entire_essay(essay_structure, context_list)

if __name__ == "__main__":
    model_name = "gpt-4o-mini-2024-07-18"  # Replace with your actual model name

    # Step 1: Generate sub-questions using InsightAnalyst
    insight_analyst = InsightAnalyst(model_name)
    research_question = "Winnie Lai argues that the seemingly innocuous act of singing 'Happy Birthday' can become a 'communal and political action' (2018: 80). Taking this as your starting point, consider the ways in which music and sound more generally have been designed and/or harnessed for the purpose of protest."
    sub_questions = generate_sub_questions(insight_analyst, research_question)
    print("Sub-Questions:", sub_questions)

    # Step 2: Structure the essay using StructuralOutlining
    outliner = SR(model_name)
    essay_structure = structure_essay(outliner, research_question, sub_questions)
    print("Essay Structure:", outliner.to_string())

    # Step 3: Iterate through the PDFs using the PDF Summary Agents & Compile A Background Info
    pdf_summary_agent = PDFSummaryAgent(model_name)
    pdf_files = [
        "/Users/wangxiang/Desktop/omnians_pro/test/demo_reading/title-all-about-iraq-re-modifying-older-slogans-and-chants-in-tishreen-october-protests-author-author-mustafa.pdf",
        "/Users/wangxiang/Desktop/omnians_pro/test/demo_reading/title-From_Soccer_Chant_to_Sonic_Meme-author-Michael_O'Brien.pdf",
        "/Users/wangxiang/Desktop/omnians_pro/test/demo_reading/title-Happy_Birthday_To_You-author-Winnie_WC_Lai.pdf",
        "/Users/wangxiang/Desktop/omnians_pro/test/demo_reading/title-On_the_Threshold_of_the_Political-author-Roshanak_Kheshti.pdf",
        "/Users/wangxiang/Desktop/omnians_pro/test/demo_reading/title-Sound_and_Movement-author-Benjamin_Tausig.pdf",
        "/Users/wangxiang/Desktop/omnians_pro/test/demo_reading/title-We're_Here!_We're_Queer_Activist-author-Mathias_Danbolt.pdf",
    ]

    background_info = ""
    for pdf_file in pdf_files:
        print(f"Summarizing PDF: {pdf_file}")
        summary = summarize_pdf(pdf_summary_agent, pdf_file)
        print(f"Summary: {summary}")
        background_info += f"\nSummary from {pdf_file}:\n{summary}\n"

    # Step 4: Use the Background Info to Revise the Structure
    print("\nCompiling background information into essay structure...")
    print(background_info)
    revisor = StructureRevisor(model_name)
    revised_structure = revise_outline(revisor, research_question, essay_structure, background_info)
    print("Revised Essay Structure:") 
    print(revised_structure)

    # Step 5: Analyze Literature using ContextAnalyst
    print("\nAnalyzing relationships between the literature...")
    literature_list = [
        "All About Iraq: Re-modifying Older Slogans and Chants in Tishreen (October) Protests",
        "From Soccer Chant to Sonic Meme",
        "Happy Birthday To You",
        "On the Threshold of the Political",
        "Sound and Movement",
        "We're Here! We're Queer Activist"
    ]
    context_analyst = ContextAnalyst(model_name)
    relationship_analysis = analyze_literature(context_analyst, background_info, literature_list)
    print("Literature Relationship Analysis:")
    print(relationship_analysis)

    # Step 6: Further Revise With Better Literature Analysis
    revised_structure_pro = revise_outline(revisor, research_question, revised_structure, relationship_analysis)
    print("First 500 characters of revised structure:")
    print(str(revised_structure_pro)[:500])

    # Step 7: Now let us write each paragraph by retrieval of evidence
    citation_retriever = Citation_Retriever()
    for pdf_path in pdf_files:
        citation_retriever.create_embedding_for_pdf(pdf_path, chunk_mode="paragraph")

    context_list = []
    for paragraph in essay_structure:
        section = paragraph.section
        purpose = paragraph.purpose
        evidence_needed = paragraph.evidence_needed
        argument_development = paragraph.argument_development
       
        print(f"Retrieving context for: {evidence_needed}")
        search_key = f"Argument to develop: {argument_development} Evidence Needed: {evidence_needed}"
        retrieved_context = retrieve_context(citation_retriever, search_key)
        context_pro = f"Answer:\n{retrieved_context['answer']}\nSources:\n{retrieved_context['context']}"
        context_list.append(context_pro)

    # Step 8: Compile the essay using ParagraphWriter
    paragraph_writer = ParagraphWriter(model_name)
    compiled_essay = compile_essay(paragraph_writer, essay_structure, context_list)

    # Assuming compiled_essay is an EssayCompilationSchema object
    combined_paragraphs = []
    combined_citations = []

    for paragraph_compilation in compiled_essay.essay:
        combined_paragraphs.append(paragraph_compilation.paragraph)
        combined_citations.extend(paragraph_compilation.references)

    print("Compiled Essay:")
    print("\n\n".join(combined_paragraphs))
    print("\nCitations:")
    print("\n".join(combined_citations))

    # Step 9: Putting Essay Together 
    whole_essay = "\n\n".join(combined_paragraphs) + "\n\n" + "\n".join(combined_citations)
    essay_compiler_agent = EssayCompiler(model_name)
    
    # Convert essay_structure to a list of dictionaries
    structure_dict = [paragraph.dict() for paragraph in essay_structure]

    better_essay = essay_compiler_agent.compile_essay(structure_dict, whole_essay)

    print("\nFinal Compiled Essay:")
    print(better_essay)

    # Optionally, save the final essay to a file
    with open("final_essay.txt", "w") as f:
        f.write(better_essay)

    print("\nFinal essay has been saved to 'final_essay.txt'")



