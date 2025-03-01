import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.schemas import SearchResultSchema
from fundations.LLMResponsePro import LLMResponsePro
from fundations.retrieval import WebRetrieval

class SearchAgent:
    def __init__(self, model_name):
        """
        Initialize SearchAgent with model name
        """
        self.llm = LLMResponsePro(model_name)
        self.web_retriever = WebRetrieval()
        self.urls_per_question = 2
        self.previous_result = None

    def search_by_subquestions(self, sub_questions: list[str]) -> dict:
        """
        Search URLs for each sub-question.
        
        Args:
            sub_questions (list[str]): List of sub-questions from InsightAnalyst
            
        Returns:
            dict: Dictionary mapping each sub-question to its search results
        """
        search_results = {}
        total_questions = len(sub_questions)
        
        print(f"\nStarting search process for {total_questions} questions...")
        
        for i, question in enumerate(sub_questions, 1):
            print(f"\n{'='*50}")
            print(f"Processing question {i}/{total_questions}")
            print(f"Question: {question}")
            print(f"{'='*50}")
            
            try:
                system_prompt = "You are a helpful assistant that searches and summarizes information."
                print(f"Searching and processing question...")
                response = self.llm.llm_output_with_search(question, system_prompt)
                
                # Print preview of the content (not the whole response dictionary)
                print(f"Response preview: {response['content'][:200]}...\n")
                search_results[question] = response
                print(f"Sources used:")
                for url in response['sources']:
                    print(f"- {url}")
                
            except Exception as e:
                print(f"Error processing question {i}: {str(e)}")
                search_results[question] = f"Error: {str(e)}"
                continue
                
        self.previous_result = search_results
        return search_results

    def to_string(self) -> str:
        """
        Convert search results to readable string format
        """
        if not self.previous_result:
            return "No search results available"
            
        output = []
        for question, result in self.previous_result.items():
            output.append(f"\nSub-question: {question}")
            if isinstance(result, dict):
                output.append(f"Answer: {result['content']}")
                output.append("\nSources:")
                for url in result['sources']:
                    output.append(f"- {url}")
            else:
                output.append(f"Answer: {result}")  # For error cases
            output.append("-" * 50)
            
        return "\n".join(output)

# Example usage
if __name__ == "__main__":
    from Agents.insightAnalyst import InsightAnalyst
    
    print("\n=== Search Agent Process Started ===")
    
    # Initialize agents
    model_name = "gpt-4o-mini-2024-07-18"
    print(f"\nInitializing agents with model: {model_name}")
    insight_analyst = InsightAnalyst(model_name)
    search_agent = SearchAgent(model_name)
    
    # Generate sub-questions
    research_question = "How has artificial intelligence impacted modern healthcare?"
    print(f"\nMain research question: {research_question}")
    
    print("\nGenerating sub-questions...")
    sub_questions = insight_analyst.generate_sub_questions(research_question)
    print("\nGenerated sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        print(f"{i}. {q}")
    
    # Search for relevant URLs
    print("\nSearching for relevant URLs...")
    search_results = search_agent.search_by_subquestions(sub_questions)
    
    # Print results
    print("\n=== Final Search Results ===")
    print(search_agent.to_string())
    
    print("\n=== Search Agent Process Completed ===")