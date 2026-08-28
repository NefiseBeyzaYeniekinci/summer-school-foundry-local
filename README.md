# 🍳 Local RAG Kitchen Assistant (Yerel RAG Mutfak Asistanı)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.62-red.svg)
![RAG](https://img.shields.io/badge/AI-RAG%20Architecture-brightgreen.svg)

This project is a **Retrieval-Augmented Generation (RAG)** based Artificial Intelligence Kitchen Assistant. It was originally developed following the methodology of the "Microsoft Foundry Local" 1-month summer school project plan, and is now fully deployed to the cloud!

**🔥 Live Demo:** [https://local-rag-assistant.streamlit.app/](https://local-rag-assistant.streamlit.app/)

The assistant combines a local vector database with a large language model (LLM) to provide highly accurate, creative, and domain-restricted culinary advice.

## ✨ Features
* **Semantic Vector Search:** Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to convert recipes into vector embeddings for meaning-based search rather than keyword matching.
* **Local Database:** Stores 500+ real recipe chunks in a lightweight, blazingly fast `SQLite` database.
* **Smart LLM Generation:** Connects to Groq's high-speed API (`gpt-oss-20b`) to generate natural, conversational responses based *strictly* on the retrieved context or its extensive culinary knowledge.
* **Anti-Hallucination Guardrails:** The AI is strictly prompt-engineered to **refuse** answering any non-food-related questions (e.g., coding, politics, math), ensuring it remains a safe and dedicated kitchen assistant.
* **Interactive UI:** A beautiful and responsive web interface built with `Streamlit`.

## 🛠️ Tech Stack
* **Language:** Python
* **UI Framework:** Streamlit
* **Embeddings:** Sentence-Transformers (HuggingFace)
* **Database:** SQLite
* **LLM Provider:** Groq (OpenAI Compatible API)
* **Data Source:** Kaggle Recipe Dataset (Pandas used for sampling)

## 🧠 Architecture Flow
1. **User asks a question** -> "What can I make with carrots and beef?"
2. **Embedder** -> Converts the text into a mathematical vector.
3. **Retriever** -> Queries `recipes.db` using Cosine Similarity to find the top 3 most relevant recipes.
4. **Generator (LLM)** -> Receives the top 3 recipes and the user's question, then formats a delicious, accurate response.

---
*Developed as a Summer School Project - 2026*
