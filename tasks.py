## For open ended
# def create_task(question):
#     return f"""
#     Conduct research and provide insights on the following question:
#     {question}
    
#     Please provide a detailed analysis and support your findings with relevant information.
#     """

## For close ended
# def create_task(input_text):
#     return f"""
#     Analyze the following input and provide relevant insights, statistics, studies:
    
#     {input_text}
    
#     Please provide a detailed analysis and support your findings with relevant information, cite links.
#     """

def create_task(input_text):
    return f"""
    Based on the following input, perform web research to find relevant links and studies that support the given topics:
    
    {input_text}
    
    For each link found, provide a summary of the content to give a synopsis before visiting the link. Use the proxy studies mentioned in the input as a guide for the type of research to be conducted.
    """