import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules (assuming OpenAI or other APIs might be used)
from openai import OpenAI
import os
from dotenv import load_dotenv
from fundations.retrieval import WebRetrieval

# Load environment variables from .env file
load_dotenv()

class LLMResponse:
    def __init__(self, model_name):
        """
        Initialize the LLMResponse with the given model name.
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
        
        if model_name == "deepseek-chat":
            self.client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
        else:
            self.client = OpenAI()
        self.retrieval = WebRetrieval()

    def llm_output(self, user_prompt, system_prompt):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return completion.choices[0].message

    def structure_output(self, schema, user_prompt, system_prompt):
        """
        Structure the output according to the provided schema, user prompt, and system prompt.
        """
        pass

    def llm_output_with_search(self, user_prompt, system_prompt):
        """
        Get LLM response with web search results
        """
        web_retriever = WebRetrieval()

        search_results = web_retriever.search_and_get_content(user_prompt)
        
        # prepare context
        context_parts = []
        for i, (content, url) in enumerate(zip(search_results["contents"], search_results["urls"])):
            context_parts.append(f"[Source {i+1}]: {url}\nContent: {content}")
        
        context = "\n\n".join(context_parts)
        enhanced_prompt = f"Context from web search:\n{context}\n\nQuestion: {user_prompt}\n\nPlease provide an answer based on the above context, and cite the source URLs in your response."
        
        response = self.llm_output(enhanced_prompt, system_prompt)
        
        # prepare response
        full_response = {
            "content": response.content,
            "sources": search_results["urls"]
        }
        
        return full_response


class Research:
    def __init__(self, research_question, background_context):
        """
        Initialize the Research object with its core properties.
        """
        self.research_question = research_question
        self.background_context = background_context
        self.skimmed_result = None  # Will be an instance of SkimmedResult
        self.outline = ""
        self.paragraphs = []
        self.citations = []
        self.first_draft = ""
        self.final_draft = ""

    def create_outline(self):
        """
        Generate an outline for the research paper based on the research question and context.
        """
        # Logic to create an outline based on the research question and background
        pass

    def draft_paper(self):
        """
        Create a draft of the research paper based on the outline and skimmed results.
        """
        # Logic to draft the paper
        pass

class SkimmedResult:
    def __init__(self, raw_reading_materials):
        """
        Initialize the SkimmedResult with raw reading materials.
        """
        self.raw_reading_materials = raw_reading_materials
        self.skimmed_graph = OmniAnsGraph()
    def skim_materials(self):
        """
        Process the raw reading materials to create a high-level knowledge graph.
        """
        # Implement the logic to skim through materials and generate a knowledge graph
        pass

class OmniAnsGraph:
    def __init__(self):
        """
        Initialize an empty OmniAnsGraph.
        """
        # Initialize graph structure or variables here
        pass

    def construct_graph(self):
        """
        Construct a knowledge graph based on the skimmed results.
        """
        # Logic to build the knowledge graph
        pass

    def search(self, query):
        """
        Search the knowledge graph for information related to a query.
        """
        # Logic to perform a search on the graph
        pass

    def concat_graph(self, additional_graph):
        """
        Concatenate another graph into this one.
        """
        # Logic to merge or concatenate graphs
        pass

# Example usage
if __name__ == "__main__":
    try:
        # # Test OpenAI
        # openai_llm = LLMResponse("gpt-4o-2024-08-06")
        # openai_response = openai_llm.llm_output(
        #     user_prompt="What is artificial intelligence?",
        #     system_prompt="You are a helpful AI expert."
        # )
        # print("\nOpenAI Response:")
        # print(openai_response.content)

        # Test DeepSeek
        deepseek_llm = LLMResponse("deepseek-chat")
        deepseek_response = deepseek_llm.llm_output(
            user_prompt="What is artificial intelligence?",
            system_prompt="You are a helpful AI expert."
        )
        print("\nDeepSeek Response:")
        print(deepseek_response.content)

        # Test OpenAI with web search
        # print("\nTesting OpenAI with web search...")
        # openai_search_response = openai_llm.llm_output_with_search(
        #     user_prompt="What are the latest developments in quantum computing?",
        #     system_prompt="You are a helpful AI expert. Please provide an up-to-date answer."
        # )
        # print("\nOpenAI Response with web search:")
        # print(openai_search_response["content"])
        # print("\nSources used:")
        # for url in openai_search_response["sources"]:
        #     print(f"- {url}")

        # Test DeepSeek with web search
        print("\nTesting DeepSeek with web search...")
        deepseek_search_response = deepseek_llm.llm_output_with_search(
            user_prompt="What are the latest developments in quantum computing?",
            system_prompt="You are a helpful AI expert. Please provide an up-to-date answer."
        )
        print("\nDeepSeek Response with web search:")
        print(deepseek_search_response["content"])
        print("\nSources used:")
        for url in deepseek_search_response["sources"]:
            print(f"- {url}")

    except Exception as e:
        print(f"Error occurred: {e}")
