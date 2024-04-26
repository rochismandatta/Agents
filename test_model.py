import os
from groq import Groq
from utils import SearchTools

# Initialize Groq client
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_haiku(prompt):
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates haikus."},
            {"role": "user", "content": prompt},
        ],
        model="llama3-70b-8192",  # Replace with a model you have access to
        max_tokens=100,
        temperature=0.7,
    )

    haiku = response.choices[0].message.content
    return haiku.strip()

# Example usage
prompt = "Tell me a haiku about modularized code."
haiku = generate_haiku(prompt)
print(haiku)