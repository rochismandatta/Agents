import os
from dotenv import load_dotenv
from agents import researcher_agent, consultant_agent, author_agent
from utils import streamlit_callback
import streamlit as st

# Load environment variables
load_dotenv()

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
                result = agent.run(task, callback=streamlit_callback)
                results.append(result)
            
            # Combine the results into a final report
            final_report = "\n\n".join(results)
            
            st.write("## Final Report")
            st.write(final_report)

if __name__ == "__main__":
    main()