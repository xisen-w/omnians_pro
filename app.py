import logging
from flask import Flask, render_template, request, jsonify
from Agents.insightAnalyst import InsightAnalyst
from Agents.structureOutliner import StructureOutliner

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    model_name = "gpt-4o-mini-2024-07-18"  # Replace with your actual model name
    research_question = request.json.get('research_question')

    if not research_question:
        app.logger.error("Research question is missing in the request")
        return jsonify({"error": "Research question is required"}), 400

    try:
        # Step 1: Generate sub-questions using InsightAnalyst
        insight_analyst = InsightAnalyst(model_name)
        sub_questions = insight_analyst.generate_sub_questions(research_question)

        # Step 2: Structure the essay using StructuralOutlining
        outliner = StructureOutliner(model_name)
        essay_structure = outliner.structure_essay(research_question, sub_questions)

        # Convert results to a readable format
        response = {
            'sub_questions': sub_questions,
            'essay_structure': outliner.previous_result.dict()
        }

        return jsonify(response)
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
