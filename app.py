import logging
from flask import Flask, render_template, request, jsonify
from Agents.insightAnalyst import InsightAnalyst
from Agents.structureOutliner import StructureOutliner
from Agents.structureRevisor import StructureRevisor
from Agents.PDFSummaryAgent import PDFSummaryAgent

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    model_name = "gpt-4o-mini-2024-07-18"  # Replace with your actual model name
    research_question = request.form.get('research_question')  # Get research question from form

    # Check if PDF files are in the request
    pdf_files = request.files.getlist('pdf_files')

    if not research_question:
        app.logger.error("Research question is missing in the request")
        return jsonify({"error": "Research question is required"}), 400

    if not pdf_files:
        app.logger.error("PDF files are missing in the request")
        return jsonify({"error": "At least one PDF file is required"}), 400

    try:
        # Step 1: Generate sub-questions using InsightAnalyst
        insight_analyst = InsightAnalyst(model_name)
        sub_questions = insight_analyst.generate_sub_questions(research_question)

        # Step 2: Structure the essay using StructureOutliner
        outliner = StructureOutliner(model_name)
        essay_structure = outliner.structure_essay(research_question, sub_questions)

        # Step 3: Summarize PDFs and compile background information
        pdf_summary_agent = PDFSummaryAgent(model_name)
        background_info = ""

        for pdf_file in pdf_files:
            app.logger.info(f"Summarizing PDF: {pdf_file.filename}")
            # Save the file temporarily and then pass its path to the PDFSummaryAgent
            pdf_path = f"/tmp/{pdf_file.filename}"
            pdf_file.save(pdf_path)  # Save file to /tmp directory
            summary = pdf_summary_agent.summarize_pdf(pdf_path=pdf_path)
            app.logger.info(f"Summary for {pdf_file.filename}: {summary}")
            background_info += f"\nSummary from {pdf_file.filename}:\n{summary}\n"

        # Step 4: Revise the structure using the background info
        revisor = StructureRevisor(model_name)
        revised_structure = revisor.revise_outline(research_question, essay_structure, background_info)

        # Response formatting: Lists should not use `.dict()`, just return them as JSON
        response = {
            'sub_questions': sub_questions,
            'initial_essay_structure': essay_structure,  # Directly send as JSON
            'revised_essay_structure': revised_structure,  # Directly send as JSON
            'background_info': background_info
        }

        return jsonify(response)
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)