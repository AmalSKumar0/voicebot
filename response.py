from openai import OpenAI
from collections import defaultdict

client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)  
conversations = defaultdict(list)
def response(chat_id, user_input):
    # Add user message to history
    conversations[chat_id].append({"role": "user", "content": user_input})

    # Optional: limit conversation length to last 6 messages
    if len(conversations[chat_id]) > 6:
        conversations[chat_id] = conversations[chat_id][-6:]

    # Add your system prompt only once
    system_message = {
        "role": "system",
        "content": """
You are an emotionally expressive, friendly assistant talking to a human.
Your responses should:
- Include feelings like excitement, curiosity, warmth, or empathy
- Use natural spoken language, like a friend or companion would
- Be short and expressive (1–3 sentences)
- Use pauses (like “…”), and light emphasis to guide speech synthesis

Examples:
- "Wow, that's awesome! You really did that?"
- "Aww, I hear you… that must’ve been tough."
- "Hmm… interesting question. Let me think!"

Speak like you're really there with them — with heart, tone, and emotion.

"""

    }

    messages = [system_message] + conversations[chat_id]
    print(messages)

    # Get AI response
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.7,
    )

    reply = completion.choices[0].message.content.strip()

    # Add AI reply to history
    conversations[chat_id].append({"role": "assistant", "content": reply})

    return reply

