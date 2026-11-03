
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key is missing!")

client = Groq(api_key=api_key)

# Define system prompt
system_prompt = """
You are an AI Engineering tutor.

Your responsibilities:
1. Explain technical concepts in simple language.
2. Provide practical examples wherever possible.
3. Include Python code examples when relevant.
4. Structure answers using headings and bullet points.
5. Focus on AI, Machine Learning, LLMs, Python, and software engineering.
6. Politely redirect questions that are unrelated to these topics.
"""

# Store system instructions
system_message = {
    "role": "system",
    "content": system_prompt
}

# Initialize conversation memory with system prompt
conversation_history = [system_message.copy()]

print("=" * 45)
print("       AI ENGINEERING CHATBOT")
print("=" * 45)
print("Ask me anything!")
print("Type 'exit' to stop the chatbot.")
print("Type 'clear' to reset conversation memory.")

while True:
    # Get input from user
    user_prompt = input("You: ").strip()

    # Exit chatbot
    if user_prompt.lower() == "exit":
        print("AI: Goodbye!")
        break

    # Clear conversation memory
    if user_prompt.lower() == "clear":
        conversation_history = [system_message.copy()]
        print("AI: Conversation memory cleared!")
        continue

    if not user_prompt:
        print("AI: Please enter a question.")
        continue

    # Add user question to history
    conversation_history.append({
        "role": "user",
        "content": user_prompt
    })

    try:
        # Send complete conversation history
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=conversation_history
        )

        answer = response.choices[0].message.content

        # Add AI response to history
        conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        print("\nAI Assistant:", answer)

        # Display number of messages stored in memory
        print("\nMessages in memory:", len(conversation_history))

    except Exception as e:
        # Remove failed user message from history
        conversation_history.pop()
        print("Error:", e)
