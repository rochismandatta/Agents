import os
import json
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from groq import Groq
from utils import SearchTools
import streamlit as st
import warnings
import time
import logging
from duckduckgo_search.exceptions import DuckDuckGoSearchException
# from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import BingSearchAPIWrapper
import nltk
nltk.download('punkt')


from dotenv import load_dotenv
load_dotenv()
import os

load_dotenv()

bing_subscription_key = os.environ["BING_SUBSCRIPTION_KEY"]
bing_search_url = os.environ["BING_SEARCH_URL"]

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Proactor event loop does not implement add_reader.*")

# Initialize Groq client
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama3-70b-8192"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Agent:
    def __init__(self, name, role, goal, tools, model, model_config):
        self.name = name
        self.role = role
        self.goal = goal
        self.tools = tools
        self.model = model
        self.model_config = model_config

    def create_task(self, input_text):
        return f"""
        Based on the following input, perform web research to find relevant links and studies that support the given topics:
        {input_text}
        For each link found, provide a summary of the content to give a synopsis after visiting the link.
        """
        # Use proxy studies which could be related to the input input as a guide for the type of research to be conducted.

    def split_query(self, query, max_length=100):
        if len(query) <= max_length:
            return [query]
        else:
            words = query.split()
            sub_queries = []
            current_query = ""
            for word in words:
                if len(current_query + " " + word) <= max_length:
                    current_query += " " + word
                else:
                    sub_queries.append(current_query.strip())
                    current_query = word
            if current_query:
                sub_queries.append(current_query.strip())
            return sub_queries

    def search_web(self, query, max_retries=3, retry_delay=1):
        if not query.strip():
            logging.info(f"Skipped searching due to empty query: '{query}'")
            return []

        logging.info(f"Searching for query: '{query}'")
        for attempt in range(max_retries):
            try:
                search = BingSearchAPIWrapper(bing_subscription_key=bing_subscription_key, bing_search_url=bing_search_url, k=3)
                results = search.results(query, num_results=3)
                if isinstance(results, list):
                    print(f"Search results for query '{query}': {results}")  # Print search results
                    return results
                else:
                    print(f"Unexpected search results format for query '{query}': {results}")  # Print unexpected format
                    return []
            except Exception as e:
                logging.error(f"Search failed (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        logging.error(f"Search failed after {max_retries} attempts.")
        return []

    def extract_example_search_query(self, search_strategy):
        # Use the LLM to extract and clean the search query
        prompt = f"""
        Given the following search strategy:
        {search_strategy}
        
        Provide a concise 20 word max cleaned search query without any additional text or explanations to be passed to a Search Engine. Write as a human would write a search query.
        """
        
        cleaned_query = self.generate_text(prompt, max_tokens=100)
        print(f"Cleaned search query: {cleaned_query}")  # Print cleaned search query
        return cleaned_query.strip()

    def extract_main_content(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            main_content = soup.find('body').get_text(strip=True)
            print(f"Extracted main content from URL '{url}': {main_content[:100]}...")  # Print extracted main content
            return main_content
        except Exception as e:
            logging.error(f"Error extracting main content from URL: {url} - {str(e)}")
            return ""

    def is_relevant(self, content, task, threshold=0.25):
        try:
            # Perform cosine similarity between the content and the task
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([content, task])
            similarity_score = cosine_similarity(vectors)[0][1]
            print(f"Similarity score between content and task: {similarity_score}")  # Print similarity score
            return similarity_score >= threshold
        except Exception as e:
            logging.error(f"Error determining relevance: {str(e)}")
            return False
    
    def summarize_text(self, text, num_sentences=10):
        try:
            # Format the prompt to instruct the model to summarize the text
            prompt = f"Summarize the following text into {num_sentences} sentences:\n\n{text}"
            
            # Use the generate_text method of the Agent class to generate the summary
            summary_text = self.generate_text(prompt, max_tokens=750)  # Adjust max_tokens as needed
            
            print(f"Generated summary: {summary_text}")  # Print generated summary
            return summary_text
        except Exception as e:
            logging.error(f"Error summarizing text with LLM: {str(e)}")
            return ""


    def generate_final_report(self, search_data):
        report = "# Final Report\n\n"
        report += "## Introduction\n"
        report += "This report presents the relevant search results and their summaries for the given task.\n\n"

        report += "## Relevant Search Results\n"
        for item in search_data:
            report += f"### {item.get('title', 'N/A')}\n"
            report += f"- URL: {item.get('link', 'N/A')}\n"
            report += f"- Summary: {item.get('summary', 'N/A')}\n\n"

        report += "## Conclusion\n"
        report += "The above search results provide relevant information and insights related to the given task."

        print(f"Generated final report: {report}")  # Print generated final report
        return report

    def run(self, task, max_iterations=3, callback=None):
        all_queries = []
        search_data = []

        # Generate initial search query
        initial_query = self.generate_text(f"Generate a search query to gather information for the task: {task}")
        example_query = self.extract_example_search_query(initial_query)
        all_queries.append(example_query)

        logging.info(f"Initial search query: {example_query}")

        for iteration in range(max_iterations):
            logging.info(f"Iteration {iteration + 1}")

            # Perform search on DuckDuckGo
            search_results = self.search_web(example_query)

            if not search_results:
                logging.info("No search results found.")
                break

            print(f"Search results: {search_results}")  # Print search results

            # Extract main content and determine relevance
            relevant_results = []
            for result in search_results:
                print(f"Result item: {result}")  # Print result item
                if isinstance(result, dict) and "link" in result:
                    main_content = self.extract_main_content(result["link"])
                    if self.is_relevant(main_content, task) and result["link"] not in [r["link"] for r in relevant_results]:
                        summary = self.summarize_text(main_content)
                        relevant_results.append({
                            "title": result.get("title", "N/A"),
                            "link": result["link"],
                            "summary": summary
                        })
                else:
                    print(f"Unexpected result format: {result}")  # Print unexpected result format

            search_data.extend(relevant_results)

            # Generate new search query based on relevant results
            if search_data:
                relevant_text = " ".join([item["summary"] for item in search_data])
                new_query = self.generate_text(f"Based on the relevant information found so far:\n\n{relevant_text}\n\nGenerate a new search query to find more relevant information for the task: {task}")
                example_query = self.extract_example_search_query(new_query)
                all_queries.append(example_query)
            else:
                logging.info("No relevant information found in this iteration.")
                break

        if not search_data:
            print("No relevant information found for the given task.")  # Print no relevant information found
            return "No relevant information found for the given task."

        # Generate final report
        final_report = self.generate_final_report(search_data)

        if callback:
            callback(final_report)

        return final_report

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
        if response_text.startswith('"') and response_text.endswith('"'):
            response_text = response_text[1:-1]
        return response_text.strip()

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

def streamlit_callback(result):
    st.write("Callback Result:", result)

def main():
    st.title("CrewAI Agents with Groq API")
    input_text = st.text_area("Enter your input:")
    if st.button("Start Research"):
        with st.spinner("Agents are working on the research..."):
            crew = [researcher_agent, consultant_agent, author_agent]
            tasks = []

            # Create tasks for each agent
            for agent in crew:
                task = agent.create_task(input_text)
                tasks.append(task)

            # Perform research and generate results
            results = []
            for task, agent in zip(tasks, crew):
                print(f"Task for {agent.name}: {task}")  # Print task for each agent
                result = agent.run(task, callback=streamlit_callback)
                results.append(result)

            # Combine the results into a final report
            final_report = "\n\n".join(results)
            print(f"Final report: {final_report}")  # Print final report
            st.write("## Final Report")
            st.write(final_report)

if __name__ == "__main__":
    main()