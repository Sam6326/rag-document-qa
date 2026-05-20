from groq import Groq
import os
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def call_llm(messages):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content

# Step 1: Extract the topic
user_input = "I want to learn about vector databases"

step1_messages = [
    {"role": "system", "content": "Extract the main technical topic from the user message. Reply with just the topic, nothing else."},
    {"role": "user", "content": user_input}
]
topic = call_llm(step1_messages)
print(f"Topic extracted: {topic}")

# Step 2: Explain it
step2_messages = [
    {"role": "system", "content": "You are a teacher. Explain the topic simply in 3 bullet points."},
    {"role": "user", "content": f"Explain: {topic}"}
]
explanation = call_llm(step2_messages)
print(f"\nExplanation:\n{explanation}")

# Step 3: Generate a quiz question
step3_messages = [
    {"role": "system", "content": "Create one quiz question to test understanding of the topic."},
    {"role": "user", "content": f"Topic: {topic}\nExplanation: {explanation}"}
]
quiz = call_llm(step3_messages)
print(f"\nQuiz question:\n{quiz}")