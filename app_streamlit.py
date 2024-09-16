import streamlit as st
import sys
import os
import logging
from utils.schemas import SubQuestionSchema, EssayStructureSchema, LiteratureRelationshipSchema

# Add this near the top of your script, after the imports
logging.basicConfig(level=logging.INFO)

# Add the parent directory to the system path
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

# Set the title of the app
st.title("Research Essay Assistant")

# Input for research question
research_question = st.text_input("Enter your research question:")

# Input for PDFs or links
pdf_links = st.text_area("Enter PDF file paths or links (one per line):")

# Choose Literature Analysis Format
analysis_format = st.multiselect(
    "Choose Literature Analysis Format(s):",
    ['Structured', 'Essay'],
    default=['Essay']
)

# Button to process the input
if st.button("Process"):
    st.write(f"Selected Literature Analysis Format(s): {', '.join(analysis_format)}")
    
    try:
        # Step 1: Generate sub-questions using InsightAnalyst
        model_name = "gpt-4o-mini-2024-07-18"  # Replace with your actual model name
        insight_analyst = InsightAnalyst(model_name)
        sub_questions = insight_analyst.generate_sub_questions(research_question)
        
        # Display sub-questions in a structured manner
        st.subheader("Sub-Questions:")
        sub_questions_schema = SubQuestionSchema(sub_questions=sub_questions)
        for i, question in enumerate(sub_questions_schema.sub_questions, 1):
            st.write(f"{i}. {question}")

    except Exception as e:
        st.error(f"An error occurred while generating sub-questions: {str(e)}")
        st.stop()

    # Split the input into a list
    pdf_files = pdf_links.splitlines()

    # Step 2: Structure the essay using StructureOutliner
    outliner = SR(model_name)
    essay_structure = outliner.structure_essay(research_question, sub_questions)
    
    # Display essay structure in a structured manner
    st.subheader("Essay Structure:")
    essay_structure_schema = EssayStructureSchema(essay_structure=essay_structure)
    for i, paragraph in enumerate(essay_structure_schema.essay_structure, 1):
        st.write(f"Paragraph {i}:")
        st.write(f"- Section: {paragraph.section}")
        st.write(f"- Purpose: {paragraph.purpose}")
        st.write(f"- Evidence Needed: {paragraph.evidence_needed}")
        st.write(f"- Argument Development: {paragraph.argument_development}")
        st.write("---")

    # Step 3: Summarize PDFs using PDF Summary Agent
    with st.spinner("Summarizing PDFs..."):
        pdf_summary_agent = PDFSummaryAgent(model_name)
        background_info = ""
        for pdf_file in pdf_files:
            try:
                st.write(f"Summarizing PDF: {pdf_file}")
                logging.info(f"Starting to summarize PDF: {pdf_file}")
                if not os.path.exists(pdf_file):
                    raise FileNotFoundError(f"PDF file not found: {pdf_file}")
                summary = pdf_summary_agent.summarize_pdf(pdf_path=pdf_file)
                st.write(f"Summary: {summary}")
                background_info += f"\nSummary from {pdf_file}:\n{summary}\n"
                logging.info(f"Successfully summarized PDF: {pdf_file}")
            except Exception as e:
                logging.error(f"Error processing PDF {pdf_file}: {str(e)}")
                st.error(f"Error processing PDF {pdf_file}: {str(e)}")
    
    # Step 4: Revise the structure using StructureRevisor
    st.subheader("Compiling Background Information...")
    st.write(background_info)
    revisor = StructureRevisor(model_name)
    revised_structure = revisor.revise_outline(research_question, essay_structure, background_info)
    st.subheader("Revised Essay Structure:")
    st.write(revised_structure)

    # Step 5: Analyze literature using ContextAnalyst
    literature_list = [pdf_file.split('/')[-1] for pdf_file in pdf_files]  # Extract titles from file paths
    context_analyst = ContextAnalyst(model_name)
    
    relationship_analysis_str = ""

    if 'Structured' in analysis_format:
        relationship_analysis = context_analyst.analyze_literature_structured(background_info, literature_list)
        st.subheader("Literature Relationship Analysis (Structured):")
        if relationship_analysis:
            for relationship in relationship_analysis:
                st.write(f"Theme: {relationship.theme}")
                st.write("Supporting Literature:")
                for support in relationship.support:
                    st.write(f"- {support}")
                st.write("Contradicting Literature:")
                for reject in relationship.reject:
                    st.write(f"- {reject}")
                st.write("Extended Literature:")
                for add_on in relationship.add_on:
                    st.write(f"- {add_on}")
                st.write("New Investigations:")
                for investigate in relationship.investigate:
                    st.write(f"- {investigate}")
                st.write("---")
        else:
            st.write("No structured analysis available.")
        
        # Convert structured analysis to string
        structured_str = "\n".join([
            f"Theme: {r.theme}\n"
            f"Supporting: {', '.join(r.support)}\n"
            f"Contradicting: {', '.join(r.reject)}\n"
            f"Extended: {', '.join(r.add_on)}\n"
            f"New Investigations: {', '.join(r.investigate)}\n"
            for r in relationship_analysis
        ])
        relationship_analysis_str += "Structured Analysis:\n" + structured_str + "\n\n"

    if 'Essay' in analysis_format:
        essay_analysis = context_analyst.analyze_literature_essay(background_info, literature_list)
        st.subheader("Literature Relationship Analysis (Essay):")
        st.write(essay_analysis)
        relationship_analysis_str += "Essay Analysis:\n" + essay_analysis

    # Check if any analysis was performed
    if not relationship_analysis_str:
        st.warning("No literature analysis format was selected. Proceeding with original structure.")
        relationship_analysis_str = "No literature analysis performed."

    # Step 6: Further Revise With Better Literature Analysis
    st.subheader("Revising structure based on literature analysis...")
    revised_structure_pro = revisor.revise_outline(research_question, revised_structure, relationship_analysis_str)
    st.subheader("Final Revised Structure:")
    st.write(revised_structure_pro)

    # Step 7: Retrieve evidence for each paragraph
    citation_retriever = Citation_Retriever()
    for pdf_path in pdf_files:
        citation_retriever.create_embedding_for_pdf(pdf_path, chunk_mode="paragraph")

    context_list = []
    for paragraph in essay_structure:
        evidence_needed = paragraph.evidence_needed
        argument_development = paragraph.argument_development
       
        st.write(f"Retrieving context for: {evidence_needed}")
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

    st.subheader("Compiled Essay:")
    st.write("\n\n".join(combined_paragraphs))
    st.subheader("Citations:")
    st.write("\n".join(combined_citations))

    # Step 9: Putting Essay Together 
    whole_essay = "\n\n".join(combined_paragraphs) + "\n\n" + "\n".join(combined_citations)
    essay_compiler_agent = EssayCompiler(model_name)
    
    # Convert essay_structure to a list of dictionaries
    structure_dict = [paragraph.dict() for paragraph in essay_structure]

    better_essay = essay_compiler_agent.compile_essay(structure_dict, whole_essay)

    st.subheader("Final Compiled Essay:")
    st.write(better_essay)

    # Step 10: Introducing Critiques
    critique_agent = CritiqueAgent(model_name)
    criticism = critique_agent.critique_essay(structure_dict, better_essay)

    st.subheader("Essay Critique:")
    st.write(criticism)

    # Step 11: Finalise the essay based on the critique
    finalise_essay_writer = FinaliseEssayWriter(model_name)
    finalised_essay = finalise_essay_writer.finalise_essay(structure_dict, better_essay, criticism)

    st.subheader("Finalised Essay:")
    st.write(finalised_essay)

    # Step 12: Save the final essay, critique, and finalised version
    with open("final_essay_complete.txt", "w") as f:
        f.write("--- Original Essay ---\n\n")
        f.write(better_essay)
        f.write("\n\n--- Essay Critique ---\n\n")
        f.write(criticism)
        f.write("\n\n--- Finalised Essay ---\n\n")
        f.write(finalised_essay)

    st.success("Final essay, critique, and finalised version have been saved to 'final_essay_complete.txt'")

# Run the app with: streamlit run app_streamlit.py