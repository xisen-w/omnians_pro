import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.insightAnalyst import InsightAnalyst 
from Agents.structureOutliner import StructureOutliner as SR
from Agents.structureRevisor import StructureRevisor
from Agents.PDFSummaryAgent import PDFSummaryAgent
from Agents.contextAnalyst import ContextAnalyst
from Agents.paragraphWriter import ParagraphWriter
from Agents.essayCompilor import EssayCompiler
from fundations.open_ai_RAG import Citation_Retriever
from Agents.critiqueAgent import CritiqueAgent
from Agents.finaliseEssayWriter import FinaliseEssayWriter

if __name__ == "__main__":
    model_name = "gpt-4o-mini-2024-07-18"  # Replace with your actual model name

    # Step 1: Generate sub-questions using InsightAnalyst
    insight_analyst = InsightAnalyst(model_name)
    research_question = "Winnie Lai argues that the seemingly innocuous act of singing 'Happy Birthday' can become a 'communal and political action' (2018: 80). Taking this as your starting point, consider the ways in which music and sound more generally have been designed and/or harnessed for the purpose of protest."
    sub_questions = insight_analyst.generate_sub_questions(research_question)
    print("Sub-Questions:", sub_questions)

    # Step 2: Structure the essay using StructuralOutlining
    outliner = SR(model_name)
    essay_structure = outliner.structure_essay(research_question, sub_questions)
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
        summary = pdf_summary_agent.summarize_pdf(pdf_path=pdf_file)
        print(f"Summary: {summary}")
        background_info += f"\nSummary from {pdf_file}:\n{summary}\n"

    # Step 4: Use the Background Info to Revise the Structure
    print("\nCompiling background information into essay structure...")
    print(background_info)
    revisor = StructureRevisor(model_name)
    revised_structure = revisor.revise_outline(research_question, essay_structure, background_info)
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
    relationship_analysis = context_analyst.analyze_literature_essay(background_info, literature_list)
    print("Literature Relationship Analysis:")
    print(relationship_analysis)

    # Step 6: Further Revise With Better Literature Analysis
    revised_structure_pro = revisor.revise_outline(research_question, revised_structure, relationship_analysis)
    print("Final Revised Structure:")
    print(revised_structure_pro)

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
        answer, context, top_texts = citation_retriever.retrieve_and_ask(search_key)
        context_pro = f"Answer:\n{answer}\nSources:\n{context}"
        context_list.append(context_pro)

    # Step 8: Compile the essay using ParagraphWriter
    paragraph_writer = ParagraphWriter(model_name)
    compiled_essay = paragraph_writer.compile_entire_essay(essay_structure, context_list)

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

    # Step 10: Introducing Critiques
    critique_agent = CritiqueAgent(model_name)
    criticism = critique_agent.critique_essay(structure_dict, better_essay)

    print("\nEssay Critique:")
    print(criticism)

    # Step 11: Save the final essay and critique
    with open("final_essay.txt", "w") as f:
        f.write(better_essay)
        f.write("\n\n--- Essay Critique ---\n\n")
        f.write(criticism)

    print("\nFinal essay and critique have been saved to 'final_essay.txt'")


    print("\nEssay Critique:")
    print(criticism)

    # Step 12: Finalise the essay based on the critique
    finalise_essay_writer = FinaliseEssayWriter(model_name)
    finalised_essay = finalise_essay_writer.finalise_essay(structure_dict, better_essay, criticism)

    print("\nFinalised Essay:")
    print(finalised_essay)

    # Step 13: Save the final essay, critique, and finalised version
    with open("final_essay_complete.txt", "w") as f:
        f.write("--- Original Essay ---\n\n")
        f.write(better_essay)
        f.write("\n\n--- Essay Critique ---\n\n")
        f.write(criticism)
        f.write("\n\n--- Finalised Essay ---\n\n")
        f.write(finalised_essay)

    print("\nFinal essay, critique, and finalised version have been saved to 'final_essay_complete.txt'")


