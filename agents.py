import asyncio
from asyncio import WindowsSelectorEventLoopPolicy
asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

import os
import json
from groq import Groq
from utils import SearchTools
import streamlit as st
# Other imports and code follow...
import warnings
# Suppress specific warnings from curl_cffi
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Proactor event loop does not implement add_reader.*")


# Initialize Groq client
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama3-8b-8192"

class Agent:
    def __init__(self, name, role, goal, tools, model, model_config):
        self.name = name
        self.role = role
        self.goal = goal
        self.tools = tools
        self.model = model
        self.model_config = model_config

    def create_task(self, input_text):
        task_prompt = f"Based on the following input, create a task for the {self.role} agent:\n\n{input_text}"
        task = self.generate_text(task_prompt)
        return task

    def search_web(self, query):
        if not query.strip():  # Check if the query is empty or just whitespace
            print(f"Skipped searching due to empty query: '{query}'")  # Debugging output
            return []  # Return an empty list if the query is empty
        print(f"Searching for query: '{query}'")  # Debugging output
        return SearchTools.search_internet(query)

    def extract_text_from_url(self, url):
        # Use the LLM to extract text from the URL
        prompt = f"Extract the main text content from the following URL: {url}"
        text = self.generate_text(prompt)
        return text

    def is_relevant(self, text, task):
        # Use the LLM to determine the relevance of the text to the task
        prompt = f"Determine if the following text is relevant to the task '{task}':\n\n{text}\n\nAnswer with 'Yes' or 'No'."
        relevance = self.generate_text(prompt)
        return "Yes" in relevance

    def summarize_text(self, text):
        # Use the LLM to generate a summary of the text
        prompt = f"Summarize the following text:\n\n{text}"
        summary = self.generate_text(prompt)
        return summary

    def run(self, task, callback=None):
        all_queries = []
        search_data = []
        
        # Generate initial search query
        initial_query = self.generate_text(f"Generate a search query to gather information for the task: {task}")
        all_queries.append(initial_query)
        
        # Perform search on DuckDuckGo
        search_results = self.search_web(initial_query)
        
        # Open and summarize relevant articles
        for i in range(3):  # Attempt up to 3 times
            relevant_results = []
            for result in search_results:
                article_text = self.extract_text_from_url(result["href"])
                if self.is_relevant(article_text, task):
                    summary = self.summarize_text(article_text)
                    relevant_results.append({
                        "title": result["title"],
                        "href": result["href"],
                        "summary": summary
                    })
            
            if relevant_results:
                search_data.extend(relevant_results)
                break
            else:
                # Generate new search query if no relevant results found
                new_query = self.generate_text(f"The search query '{initial_query}' did not yield relevant results for the task '{task}'. Generate a new search query to find more relevant information.")
                all_queries.append(new_query)
                search_results = self.search_web(new_query)
        
        if not search_data:
            return "No relevant information found for the given task."
        
        # Generate report
        report_prompt = f"Analyze the following search data and generate a comprehensive report on the task '{task}':\n\n{json.dumps(search_data)}"
        report = self.generate_text(report_prompt, max_tokens=4000)
        
        if callback:
            callback(report)
        
        return report

    def generate_text(self, prompt, model=None, max_tokens=2000, temperature=None):
        if model is None:
            model = self.model

        if temperature is None:
            temperature = self.model_config.get("temperature", 0.7)

        # Prepare model configuration
        model_config = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self.model_config,
        }
        model_config.pop("model", None)
        model_config.pop("max_length", None)

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"You are a {self.role} with the goal: {self.goal}"},
                {"role": "user", "content": prompt},
            ],
            model=model,
            **model_config
        )

        response_text = response.choices[0].message.content
        # Remove outer double quotes if present
        if response_text.startswith('"') and response_text.endswith('"'):
            response_text = response_text[1:-1] 
        return response_text.strip()  # Return the response text as is, without json.dumps()


# Define agents
researcher_agent = Agent(
    name="Researcher",
    role="Senior research analyst",
    goal="Conduct comprehensive analysis and provide strategic insights.",
    tools=[SearchTools.search_internet],
    model=MODEL,
    model_config={
        "model": MODEL,
        "max_length": 8192,
        "temperature": 0.7,
    },
)

consultant_agent = Agent(
    name="Consultant",
    role="Business Angel and venture capital consultant",
    goal="Provide funding, mentorship, and strategic guidance to startups.",
    tools=[SearchTools.search_internet],
    model=MODEL,
    model_config={
        "model": MODEL,
        "max_length": 8192,
        "temperature": 0.7,
    },
)

author_agent = Agent(
    name="Author",
    role="Tech content author",
    goal="Create high-quality content on technology topics.",
    tools=[],
    model=MODEL,
    model_config={
        "model": MODEL,
        "max_length": 8192,
        "temperature": 0.7,
    },
)

def test_search_web():
    query = "Apple's success in the luxury watch market"
    search_results = researcher_agent.search_web(query)
    assert len(search_results) > 0, "Expected search results, but got none"
    for result in search_results:
        assert "title" in result, "Expected 'title' key in search result"
        assert "href" in result, "Expected 'href' key in search result"
        assert "body" in result, "Expected 'body' key in search result"

# Run the test case
# test_search_web()