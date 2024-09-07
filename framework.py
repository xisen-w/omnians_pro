from Agents.insightAnalyst import InsightAnalyst
from Agents.structureOutliner import StructureOutliner as SR
from Agents.structureRevisor import StructureRevisor
from Agents.PDFSummaryAgent import PDFSummaryAgent

if __name__ == "__main__":
    model_name = "gpt-4o-mini-2024-07-18"  # Replace with your actual model name

    # Step 1: Generate sub-questions using InsightAnalyst
    insight_analyst = InsightAnalyst(model_name)
    research_question = "What is a good musical?"
    sub_questions = insight_analyst.generate_sub_questions(research_question)
    print("Sub-Questions:", sub_questions)

    # Step 2: Structure the essay using StructuralOutlining
    outliner = SR(model_name)
    essay_structure = outliner.structure_essay(research_question, sub_questions)
    print("Essay Structure:", outliner.to_string())

    # Step 3: Iterate through the pdfs using the PDF Summary Agents & Compile A Background Info

    # Use the Background Info to Revise the Stcuture by distilling strong information from the background materials 

    # Initialize PDFSummaryAgent
    pdf_summary_agent = PDFSummaryAgent(model_name)

    # List of PDF paths or URLs to summarize (you can add more)
    pdf_files = [
        "/Users/wangxiang/Desktop/omnians_pro/test/readings/science1.pdf",  # Replace with actual paths
        "/Users/wangxiang/Desktop/omnians_pro/test/readings/socialSci1.pdf"
    ]

    background_info = ""  # This will hold the compiled background info

    # Iterate over the PDFs and summarize them
    for pdf_file in pdf_files:
        print(f"Summarizing PDF: {pdf_file}")
        summary = pdf_summary_agent.summarize_pdf(pdf_path=pdf_file)
        print(f"Summary: {summary}")
        background_info += f"\nSummary from {pdf_file}:\n{summary}\n"

    # Step 4: Use the Background Info to Revise the Structure
    print("\nCompiling background information into essay structure...")

    revisor = StructureRevisor(model_name)
    
    # Here you can revise the structure using background info
    # Assume that you have a method `revise_structure` to revise the structure based on the background info
    revised_structure = revisor.revise_outline(research_question, essay_structure, background_info)
    
    print("Revised Essay Structure:")
    print(outliner.to_string())


    







