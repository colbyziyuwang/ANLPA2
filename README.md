# 🕵️‍♂️ Nezha Buys, Tang Seng Waits: AI-Powered SEC Filing Detective

A Streamlit-based AI web application for analyzing SEC filings (10-K and DEF 14A), predicting stock price movements, and making it fun with iconic characters like Nezha, Tang Seng, and Elon Musk. The app supports both Retrieval-Augmented Generation (RAG) and predictive models trained on financial data and document embeddings.

---

## 🚀 Features

- 🔍 **Search and Analyze SEC Filings** (10-K / DEF 14A)
- 🧠 **Choose AI Model** (LLM-based RAG or GRU-based classifier)
- 📊 **Predict Stock Movement** using:
  - Past 7-day stock data
  - Economic indicators (CPI, Inflation, etc.)
  - Optional: Document embeddings via Doc2Vec
- 🧬 **Model Options**:
  - `stock_gru_model.pth`: Stock + economic indicators only
  - `stock_gru_d2v.pth`: Includes SEC document embeddings
- 🧠 **RAG Chatbot Integration** with Ollama + HuggingFace Embeddings
- 🎭 **Fun Character Picker** with risk scores from 0 to 1
- 📈 **7-Day Price Trend Visualizations**

---
