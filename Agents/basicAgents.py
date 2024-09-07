from abc import ABC, abstractmethod
from fundations.LLMResponsePro import LLMResponsePro
from fundations.foundation import LLMResponse
from pydantic import BaseModel, Field

class Agent(ABC):
    def __init__(self):
        self.previous_result = None

    @abstractmethod
    def perform_action(self, **kwargs):
        """
        Abstract method to perform an action. Must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def to_string(self, **kwargs):  # Convert the result to a string for printing
        """
        Abstract method to convert the result to a string. Must be implemented by all subclasses.
        """
        pass

    def use_previous_result(self):
        """
        Access the result from the previous action, which can be used for subsequent actions.
        """
        return self.previous_result

class LLMAgent(Agent):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.llm_pro = LLMResponsePro(model_name)
        self.llm = LLMResponse(model_name)

    def perform_action(self, user_prompt: str, system_prompt: str, schema_class: BaseModel = None):
        """
        Perform an action using the LLM with explicit parameters.
        Use LLMResponsePro if schema_class is provided, otherwise use LLMResponse.
        """
        if schema_class:
            response = self.llm_pro.structured_output(
                schema_class=schema_class,
                user_prompt=user_prompt,
                system_prompt=system_prompt
            )
        else:
            response = self.llm.llm_output(
                user_prompt=user_prompt,
                system_prompt=system_prompt
            )
        self.previous_result = response
        return response

    def to_string(self):
        """
        Convert the LLM response to a readable string.
        """
        if self.previous_result:
            if isinstance(self.previous_result, BaseModel):
                return self.previous_result.json(indent=2)
            return str(self.previous_result)
        return "No result available."


class SearchAgent(Agent):
    def __init__(self, search_engine):
        super().__init__()
        self.search_engine = search_engine

    def perform_action(self, query, **kwargs):
        """
        Perform a search action using the provided search engine.
        """
        results = self.search_engine.search(query)
        self.previous_result = results
        return results

    def to_string(self):
        """
        Convert the search results to a readable string.
        """
        if self.previous_result:
            # Convert the results into a formatted string (assuming it's a list of results)
            return "\n".join([str(result) for result in self.previous_result])
        return "No result available."
