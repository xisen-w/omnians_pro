import streamlit as st
import sys
import os
import logging
from utils.schemas import SubQuestionSchema, EssayStructureSchema, LiteratureRelationshipSchema, ParagraphSchema
import tempfile
import re

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

# Add this near the top of your script, after the imports
logging.basicConfig(level=logging.INFO)

# Helper function to get filename from path
def get_filename(path):
    return original_file_names.get(path, os.path.basename(path))

# Set page config for wide layout
st.set_page_config(layout="wide")

# Initialize session state
if 'sub_questions' not in st.session_state:
    st.session_state.sub_questions = None
if 'essay_structure' not in st.session_state:
    st.session_state.essay_structure = None
if 'background_info' not in st.session_state:
    st.session_state.background_info = ""
if 'revised_structure' not in st.session_state:
    st.session_state.revised_structure = None
if 'relationship_analysis' not in st.session_state:
    st.session_state.relationship_analysis = None
if 'final_structure' not in st.session_state:
    st.session_state.final_structure = None
if 'context_list' not in st.session_state:
    st.session_state.context_list = None
if 'compiled_essay' not in st.session_state:
    st.session_state.compiled_essay = None
if 'criticism' not in st.session_state:
    st.session_state.criticism = None
if 'finalised_essay' not in st.session_state:
    st.session_state.finalised_essay = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# Create a sidebar for navigation
st.sidebar.title("Navigation")
steps = [
    "Research Question",
    "Sub-Questions",
    "Initial Essay Structure",
    "Document Summaries",
    "Revised Essay Structure",
    "Literature Analysis",
    "Final Revised Structure",
    "Evidence Retrieval",
    "Essay Compilation",
    "Essay Critique",
    "Final Essay"
]

# Navigation in sidebar
selected_step = st.sidebar.radio("Go to Step", steps, index=st.session_state.current_step)

# Update current_step based on sidebar selection
st.session_state.current_step = steps.index(selected_step)

# Main content
st.title("Research Essay Assistant")

# Input for research question
research_question = st.text_input("Enter your research question:")

# Input for documents or links
doc_links = st.text_area("Enter document URLs or local file paths (one per line or separated by semicolons):")

# File uploader for documents
uploaded_files = st.file_uploader("Or upload document files", type=["pdf", "txt", "doc", "docx"], accept_multiple_files=True)

# Choose Literature Analysis Format
analysis_format = st.multiselect(
    "Choose Literature Analysis Format(s):",
    ['Structured', 'Essay'],
    default=['Essay']
)

# Create columns for Automate and Co-pilot buttons
col1, col2 = st.columns(2)

# Automate button
if col1.button("OmniAns Your Answer"):
    st.session_state.current_step = len(steps) - 1  # Set to last step to run all
    st.rerun()  # Force a rerun to update the sidebar

# # Co-pilot button
# if col2.button("Next Step"):
#     st.session_state.current_step = min(st.session_state.current_step + 1, len(steps) - 1)
#     st.rerun()  # Force a rerun to update the sidebar

# Display current step
st.write(f"Current Step: {steps[st.session_state.current_step]}")

# Main process
try:
    model_name = "gpt-4o-mini-2024-07-18"  # Replace with your actual model name

    # Process document links and uploaded files
    doc_sources = []
    if doc_links:
        doc_sources = [item.strip() for sublist in [re.split(r'[;\n]', line) for line in doc_links.split('\n')] for item in sublist if item.strip()]
    
    # Create temporary files for uploaded documents
    temp_files = []
    original_file_names = {}
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_files.append(tmp_file.name)
            original_file_names[tmp_file.name] = uploaded_file.name
    
    doc_sources.extend(temp_files)

    # Step 1: Generate sub-questions using InsightAnalyst
    if st.session_state.current_step >= 1:
        st.header("🧠 Insight Analyst")
        if st.session_state.sub_questions is None:
            insight_analyst = InsightAnalyst(model_name)
            st.session_state.sub_questions = insight_analyst.generate_sub_questions(research_question)
        
        # Display sub-questions in a structured manner
        sub_questions_schema = SubQuestionSchema(sub_questions=st.session_state.sub_questions)
        for i, question in enumerate(sub_questions_schema.sub_questions, 1):
            st.write(f"{i}. {question}")

    # Step 2: Structure the essay using StructureOutliner
    if st.session_state.current_step >= 2:
        st.header("📝 Structure Outliner")
        if st.session_state.essay_structure is None:
            outliner = SR(model_name)
            st.session_state.essay_structure = outliner.structure_essay(research_question, st.session_state.sub_questions)
        
        # Display essay structure in a structured manner
        essay_structure_schema = EssayStructureSchema(essay_structure=st.session_state.essay_structure)
        for i, paragraph in enumerate(essay_structure_schema.essay_structure, 1):
            st.subheader(f"Paragraph {i}:")
            st.write(f"- Section: {paragraph.section}")
            st.write(f"- Purpose: {paragraph.purpose}")
            st.write(f"- Evidence Needed: {paragraph.evidence_needed}")
            st.write(f"- Argument Development: {paragraph.argument_development}")
            st.write("---")

    # Step 3: Summarize documents using Summary Agent
    if st.session_state.current_step >= 3:
        st.header("📚 Summary Agent")
        if not st.session_state.background_info:
            with st.spinner("Summarizing documents..."):
                summary_agent = PDFSummaryAgent(model_name)
                for doc_source in doc_sources:
                    try:
                        doc_name = get_filename(doc_source)
                        st.subheader(f"Summarizing document: {doc_name}")
                        logging.info(f"Starting to summarize document: {doc_name}")
                        summary = summary_agent.summarize_pdf(pdf_path=doc_source)
                        st.write(summary)
                        st.session_state.background_info += f"\nSummary from {doc_name}:\n{summary}\n"
                        logging.info(f"Successfully summarized document: {doc_name}")
                    except Exception as e:
                        logging.error(f"Error processing document {doc_name}: {str(e)}")
                        st.error(f"Error processing document {doc_name}: {str(e)}")
        else:
            st.write(st.session_state.background_info)
    
    # Step 4: Revise the structure using StructureRevisor
    if st.session_state.current_step >= 4:
        st.header("🔄 Structure Revisor")
        st.subheader("Background Information:")
        st.write(st.session_state.background_info)
        if st.session_state.revised_structure is None:
            revisor = StructureRevisor(model_name)
            st.session_state.revised_structure = revisor.revise_outline(research_question, st.session_state.essay_structure, st.session_state.background_info).essay_structure
        
        # Display revised structure in a structured manner
        try:
            revised_structure_schema = EssayStructureSchema(essay_structure=st.session_state.revised_structure)
            for i, paragraph in enumerate(revised_structure_schema.essay_structure, 1):
                st.subheader(f"Paragraph {i}:")
                st.write(f"- Section: {paragraph.section}")
                st.write(f"- Purpose: {paragraph.purpose}")
                st.write(f"- Evidence Needed: {paragraph.evidence_needed}")
                st.write(f"- Argument Development: {paragraph.argument_development}")
                st.write("---")
        except Exception as e:
            st.error(f"Error displaying revised structure: {str(e)}")
            logging.error(f"Error displaying revised structure: {str(e)}", exc_info=True)

    # Step 5: Analyze literature using ContextAnalyst
    if st.session_state.current_step >= 5:
        st.header("📊 Context Analyst")
        if st.session_state.relationship_analysis is None:
            literature_list = [get_filename(doc_file) for doc_file in doc_sources]
            context_analyst = ContextAnalyst(model_name)
            
            relationship_analysis_str = ""

            if 'Structured' in analysis_format:
                st.subheader("Structured Analysis:")
                relationship_analysis = context_analyst.analyze_literature_structured(st.session_state.background_info, literature_list)
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
                st.subheader("Essay Analysis:")
                essay_analysis = context_analyst.analyze_literature_essay(st.session_state.background_info, literature_list)
                st.write(essay_analysis)
                relationship_analysis_str += "Essay Analysis:\n" + essay_analysis

            st.session_state.relationship_analysis = relationship_analysis_str
        else:
            st.write(st.session_state.relationship_analysis)

    # Step 6: Further Revise With Better Literature Analysis
    if st.session_state.current_step >= 6:
        st.header("🔄 Structure Revisor Pro")
        if st.session_state.final_structure is None:
            revisor = StructureRevisor(model_name)
            st.session_state.final_structure = revisor.revise_outline(research_question, st.session_state.revised_structure, st.session_state.relationship_analysis).essay_structure
        
        # Display final revised structure in a structured manner
        final_structure_schema = EssayStructureSchema(essay_structure=st.session_state.final_structure)
        for i, paragraph in enumerate(final_structure_schema.essay_structure, 1):
            st.subheader(f"Paragraph {i}:")
            st.write(f"- Section: {paragraph.section}")
            st.write(f"- Purpose: {paragraph.purpose}")
            st.write(f"- Evidence Needed: {paragraph.evidence_needed}")
            st.write(f"- Argument Development: {paragraph.argument_development}")
            st.write("---")

    # Step 7: Retrieve evidence for each paragraph
    if st.session_state.current_step >= 7:
        st.header("🔍 Citation Retriever")
        if st.session_state.context_list is None:
            citation_retriever = Citation_Retriever()
            for doc_path in doc_sources:
                doc_name = get_filename(doc_path)
                st.write(f"Creating embedding for: {doc_name}")
                citation_retriever.create_embedding_for_pdf(doc_path, chunk_mode="paragraph")

            st.session_state.context_list = []
            for paragraph in st.session_state.final_structure:
                evidence_needed = paragraph.evidence_needed
                argument_development = paragraph.argument_development
               
                st.subheader(f"Retrieving context for: {evidence_needed}")
                search_key = f"Argument to develop: {argument_development} Evidence Needed: {evidence_needed}"
                answer, context, top_texts = citation_retriever.retrieve_and_ask(search_key)
                context_pro = f"Answer:\n{answer}\nSources:\n{context}"
                st.session_state.context_list.append(context_pro)
                st.write(context_pro)
        else:
            for i, context in enumerate(st.session_state.context_list, 1):
                st.subheader(f"Context for Paragraph {i}:")
                st.write(context)

    # Step 8: Compile the essay using ParagraphWriter
    if st.session_state.current_step >= 8:
        st.header("✍️ Paragraph Writer")
        if st.session_state.compiled_essay is None:
            paragraph_writer = ParagraphWriter(model_name)
            st.session_state.compiled_essay = paragraph_writer.compile_entire_essay(st.session_state.final_structure, st.session_state.context_list)

        combined_paragraphs = []
        combined_citations = []

        for paragraph_compilation in st.session_state.compiled_essay.essay:
            combined_paragraphs.append(paragraph_compilation.paragraph)
            combined_citations.extend(paragraph_compilation.references)

        st.subheader("Compiled Essay:")
        st.write("\n\n".join(combined_paragraphs))
        st.subheader("Citations:")
        st.write("\n".join(combined_citations))

    # Step 9: Critique the Essay
    if st.session_state.current_step >= 9:
        st.header("🧐 Critique Agent")
        if st.session_state.criticism is None:
            whole_essay = "\n\n".join(combined_paragraphs) + "\n\n" + "\n".join(combined_citations)
            essay_compiler_agent = EssayCompiler(model_name)
            better_essay = essay_compiler_agent.compile_essay([paragraph.dict() for paragraph in st.session_state.final_structure], whole_essay)

            critique_agent = CritiqueAgent(model_name)
            st.session_state.criticism = critique_agent.critique_essay([paragraph.dict() for paragraph in st.session_state.final_structure], better_essay)

        st.subheader("Essay Critique:")
        st.write(st.session_state.criticism)

    # Step 10: Finalize the essay based on the critique
    if st.session_state.current_step >= 10:
        st.header("📝 Final Essay Writer")
        if st.session_state.finalised_essay is None:
            finalise_essay_writer = FinaliseEssayWriter(model_name)
            st.session_state.finalised_essay = finalise_essay_writer.finalise_essay([paragraph.dict() for paragraph in st.session_state.final_structure], better_essay, st.session_state.criticism)

        st.subheader("Finalised Essay:")
        st.write(st.session_state.finalised_essay)

        # Save the final essay, critique, and finalised version
        with open("final_essay_complete.txt", "w") as f:
            f.write("--- Original Essay ---\n\n")
            f.write(better_essay)
            f.write("\n\n--- Essay Critique ---\n\n")
            f.write(st.session_state.criticism)
            f.write("\n\n--- Finalised Essay ---\n\n")
            f.write(st.session_state.finalised_essay)

        st.success("Final essay, critique, and finalised version have been saved to 'final_essay_complete.txt'")

    # Clean up temporary files
    for temp_file in temp_files:
        os.unlink(temp_file)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    logging.error(f"Error: {str(e)}", exc_info=True)
    st.stop()

# Run the app with: streamlit run app_streamlit.py