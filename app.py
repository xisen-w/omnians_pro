from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from Agents.insightAnalyst import InsightAnalyst
from Agents.structureOutliner import StructureOutliner
from Agents.PDFSummaryAgent import PDFSummaryAgent
from Agents.structureRevisor import StructureRevisor

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class InvitationCode(Base):
    __tablename__ = "invitation_codes"

    code = Column(String, primary_key=True, index=True)
    is_valid = Column(Boolean, default=True)

Base.metadata.create_all(bind=engine)

# Authentication function
def authenticate(invitation_code):
    db = SessionLocal()
    try:
        code = db.query(InvitationCode).filter(InvitationCode.code == invitation_code, InvitationCode.is_valid == True).first()
        return code is not None
    finally:
        db.close()

@app.route('/api/authenticate', methods=['POST'])
def auth_route():
    data = request.json
    invitation_code = data.get('invitation_code')
    if authenticate(invitation_code):
        return jsonify({"success": True, "message": "Authentication successful"})
    else:
        return jsonify({"success": False, "message": "Invalid invitation code"}), 401

@app.route('/api/analyze', methods=['POST'])
def analyze_route():
    data = request.json
    research_question = data.get('research_question')
    pdf_files = data.get('pdf_files')  # This should be a list of file URLs or content

    if not research_question or not pdf_files:
        return jsonify({"error": "Missing research question or PDF files"}), 400

    try:
        result = analyze(research_question, pdf_files)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def analyze(research_question, pdf_files):
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini-2024-07-18")

    # Step 1: Generate sub-questions
    insight_analyst = InsightAnalyst(model_name)
    sub_questions = insight_analyst.generate_sub_questions(research_question)

    # Step 2: Structure the essay
    outliner = StructureOutliner(model_name)
    essay_structure = outliner.structure_essay(research_question, sub_questions)

    # Step 3: Summarize PDFs and compile background information
    pdf_summary_agent = PDFSummaryAgent(model_name)
    background_info = ""

    for pdf_file in pdf_files:
        # You'll need to implement a way to download or access the PDF content here
        summary = pdf_summary_agent.summarize_pdf(pdf_file)
        background_info += f"\nSummary from {pdf_file}:\n{summary}\n"

    # Step 4: Revise the structure using the background info
    revisor = StructureRevisor(model_name)
    revised_structure = revisor.revise_outline(research_question, essay_structure, background_info)

    # Prepare response
    response = {
        'sub_questions': sub_questions,
        'initial_essay_structure': essay_structure,
        'revised_essay_structure': revised_structure,
        'background_info': background_info
    }

    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)