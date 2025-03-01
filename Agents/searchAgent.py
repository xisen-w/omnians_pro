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
        
        for question in sub_questions:
            # Get initial search results
            search_data = self.web_retriever.search_and_get_content(question)
            urls = search_data["urls"]
            contents = search_data["contents"]
            
            # Validate and filter results
            validated_urls = []
            for url, content in zip(urls, contents):
                # Prepare prompt for relevance check
                prompt = f"""
                Please evaluate if this content is relevant to the question.
                
                Question: {question}
                
                Content from {url}:
                {content[:500]}...
                
                Evaluate the relevance and provide a score.
                """
                
                # Get LLM evaluation
                response = self.llm.perform_action(
                    system_prompt="You are an expert at evaluating content relevance.",
                    schema_class=SearchResultSchema,
                    user_prompt=prompt
                )
                
                if response and response.is_relevant:
                    validated_urls.append({
                        'url': url,
                        'relevance_score': response.relevance_score
                    })
            
            # Sort by relevance score and take top 2
            validated_urls.sort(key=lambda x: x['relevance_score'], reverse=True)
            search_results[question] = validated_urls[:self.urls_per_question]
        
        self.previous_result = search_results
        return search_results

    def to_string(self) -> str:
        """
        Convert search results to readable string format
        """
        if not self.previous_result:
            return "No search results available"
            
        output = []
        for question, urls in self.previous_result.items():
            output.append(f"Sub-question: {question}")
            for i, url_info in enumerate(urls, 1):
                output.append(f"  URL {i}: {url_info['url']}")
                output.append(f"  Relevance Score: {url_info['relevance_score']:.2f}")
            output.append("")
            
        return "\n".join(output)

# Example usage
if __name__ == "__main__":
    from Agents.insightAnalyst import InsightAnalyst
    
    # Initialize agents
    model_name = "gpt-4o-mini-2024-07-18"
    insight_analyst = InsightAnalyst(model_name)
    search_agent = SearchAgent(model_name)
    
    # Generate sub-questions
    research_question = "How has artificial intelligence impacted modern healthcare?"
    sub_questions = insight_analyst.generate_sub_questions(research_question)
    
    # Search for relevant URLs
    search_results = search_agent.search_by_subquestions(sub_questions)
    
    # Print results
    print("Search Results:")
    print(search_agent.to_string())