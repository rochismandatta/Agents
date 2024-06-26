CrewAI Agents with Groq API
==========================

CrewAI Agents is a powerful research tool that leverages AI agents to conduct comprehensive analysis and generate strategic insights based on user input. This project combines the capabilities of the Groq API, DuckDuckGo search, and Streamlit to create an interactive web application.

Features
--------

* Three specialized AI agents: Researcher, Consultant, and Author
* Agents generate tasks based on user input and perform web searches to gather relevant information
* Agents extract text from articles, determine relevance to the task, and generate summaries
* Multiple search iterations to find the most relevant information
* Final report generated by combining the results from all agents
* User-friendly web interface built with Streamlit

Installation
------------

### Clone the repository:

```bash
git clone https://github.com/rochismandatta/Agents.git
cd Agents
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```
Set up the environment variables:
Create a .env file in the project root and add your Groq API key:

* GROQ_API_KEY=your-api-key
* BING_SUBSCRIPTION_KEY = your-subscription_key
* BING_SEARCH_URL = 'https://api.bing.microsoft.com/v7.0/search'
### Usage
Run the Streamlit application:
streamlit run app.py
Open your web browser and navigate to http://localhost:8501. Enter your input in the text area and click the "Start Research" button. The AI agents will work together to generate a comprehensive report based on your input.

## Notes:
Implementation with DuckDuckGo search engine for agents.py and Bing Search engine API for rec_res.py
```bash
streamlit run .\rec_res.py
```

## AI Agents
### Researcher Agent
Role: Senior research analyst
Goal: Conduct comprehensive analysis and provide strategic insights
### Consultant Agent
Role: Business Angel and venture capital consultant
Goal: Provide funding, mentorship, and strategic guidance to startups
### Author Agent
Role: Tech content author
Goal: Create high-quality content on technology topics
Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgements
Groq API for providing the AI capabilities
DuckDuckGo for the search functionality
Streamlit for the web application framework
