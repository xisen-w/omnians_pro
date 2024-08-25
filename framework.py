from Agents.insightAnalyst import InsightAnalyst
from Agents.structureOutliner import StructureOutliner

if __name__ == "__main__":
    model_name = "gpt-4o-mini-2024-07-18"  # Replace with your actual model name

    # Step 1: Generate sub-questions using InsightAnalyst
    insight_analyst = InsightAnalyst(model_name)
    research_question = "What is a good musical?"
    sub_questions = insight_analyst.generate_sub_questions(research_question)
    print("Sub-Questions:", sub_questions)

    # Step 2: Structure the essay using StructuralOutlining
    outliner = StructureOutliner(model_name)
    essay_structure = outliner.structure_essay(research_question, sub_questions)
    print("Essay Structure:", outliner.to_string())







