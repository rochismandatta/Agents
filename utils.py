import streamlit as st
from duckduckgo_search import DDGS

class SearchTools:
    @staticmethod
    def search_internet(query):
        search_results = []
        results = DDGS().text(query)
        for result in results:
            search_results.append({
                "title": result["title"],
                "href": result["href"],
                "body": result["body"]
            })
        return search_results

def streamlit_callback(result):
    st.write(result)