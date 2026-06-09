transcripts = [
    {
        "session": "Handle stress better",
        "text": "User said they feel very stressed at work and overwhelmed by deadlines. Coach suggested breathing exercises and setting boundaries. User committed to taking a 10 minute walk every day."
    },
    {
        "session": "Get clearer",
        "text": "User is unsure about their career direction. They are interested in tech but lack confidence. User mentioned wanting to apply for internships on LinkedIn and update their resume."
    },
    {
        "session": "Set realistic goals",
        "text": "User wants to build better habits. They struggle with procrastination. Coach helped user set a goal of studying for 2 hours every morning before checking their phone."
    }
]

print("Transcripts loaded:", len(transcripts))

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

chunks=[]
chunks = []
for transcript in transcripts:
    chunks.append({
        "session": transcript["session"],
        "text": transcript["text"]
    })

texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts)

print("Chunks embedded:", len(chunks))

import numpy as np

question = "What has the user committed to doing about stress?"

question_embedding = model.encode(question)

similarities = np.dot(embeddings, question_embedding)

best_index = np.argmax(similarities)

print("Most relevant session:", chunks[best_index]["session"])
print("Content:", chunks[best_index]["text"])

import anthropic

client = anthropic.Anthropic(api_key="ANTHROPIC_API_KEY")

context = chunks[best_index]["text"]

message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[
        {
            "role": "user",
            "content": f"Based on this coaching session transcript:\n\n{context}\n\nAnswer this question: {question}"
        }
    ]
)

print("\nClaude's answer:")
print(message.content[0].text)
