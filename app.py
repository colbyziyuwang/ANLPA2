import streamlit as st
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceWindowNodeParser
import tempfile
import os
import matplotlib.pyplot as plt
import pandas as pd

# ✅ Streamlit UI
st.title("💸 AI-Powered SEC Filing Detective 🕵️‍♂️")
st.write("Upload a 10-K filing and let this chatbot do the reading for you. Just ask: 'Will this stock 🚀 or 📉?'")

# ✅ About
st.write("### ℹ️ About")
st.write("Welcome to the **SEC Filing Analysis RAG Chatbot** where serious finance meets questionable humor. 🚀")

st.write("This chatbot is designed to read 10-K filings so you don’t have to. Ask it why a stock might go 🚀 or 📉 after reading financial statements.")

st.write("#### 🎭 Pick a Character for Fun")
character_name = st.text_input("Enter your character name:")

if st.button("Pick a Character"):
    character_name = character_name.strip().lower()
    
    character_responses = {
        "spongebob": "Spongebob is a yellow sponge who lives in a pineapple under the sea. \nAlso, he works for free at the Krusty Krab—just like an unpaid intern.",
        "patrick": "Patrick Star is a professional rock dweller. His financial advice? 'Just don’t spend money.' Genius. 💰",
        "nezha": "我命由我不由天！ (Translation: I control my own destiny, not the heavens! Also, I refuse to pay taxes.)",
        "elon musk": "Elon Musk: Likes rockets, AI, and tweeting at 3 AM. SEC filings? He prefers making headlines instead. 🚀",
        "batman": "Batman doesn’t read SEC filings—he **owns** the companies filing them. Also, no parents. 😢",
        "rick sanchez": "Rick: '10-K filings? Pfft. Just invest in interdimensional markets, Morty!' 🤯",
        "shrek": "Shrek: 'This chatbot is like an onion—it has layers. Also, I don’t do stocks, I do **swamps**.' 🧅",
        "default": "Hmm... I don't know that character. Maybe they’re off trading crypto?"
    }
    
    st.write(character_responses.get(character_name, character_responses["default"]))

# ✅ Load the LLM Model
llm = Ollama(
    model="llama3.2", context_window=4096, request_timeout=600.0, temperature=0.1
)

# ✅ Load the embedding model
embedding_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ Configure Settings
Settings.llm = llm
Settings.embed_model = embedding_model

# ✅ File Upload
uploaded_file = st.file_uploader("Upload 10-K filing (TXT format)", type=["txt"])

if uploaded_file is not None:
    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    # ✅ Load documents
    docs = SimpleDirectoryReader(input_files=[temp_file_path]).load_data()
    
    # ✅ Create Node Parser with Sentence Window
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=1, window_metadata_key="window", original_text_metadata_key="original_text"
    )
    
    # ✅ Process nodes from documents
    nodes = node_parser.get_nodes_from_documents(docs)
    
    # ✅ Create Vector Store Index
    index = VectorStoreIndex(nodes)
    
    # ✅ Create Retriever
    retriever = index.as_retriever(similarity_top_k=3)
    
    # ✅ Create Query Engine
    query_engine = RetrieverQueryEngine(retriever=retriever)
    
    # ✅ Chat UI
    user_query = st.text_input("Ask a question about the SEC filing:", "Give me potential reasons the stock will go up or down after reading this file.")
    
    if st.button("Analyze SEC Filing"):
        response = query_engine.query(user_query)
        st.subheader("📊 RAG Response:")
        st.write(response.response)
    
    # Clean up temporary file
    os.remove(temp_file_path)

# ✅ Stock Price Analysis
uploaded_stock_file = st.file_uploader("Upload Stock CSV File", type=["csv"])

if uploaded_stock_file is not None:
    date_input = st.text_input("Enter the date of the stock (YYYY-MM-DD):")
    
    if st.button("Analyze Stock") and date_input:
        st.write("We will plot the 7-day price of the stock starting from the date you entered.")
        
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(uploaded_stock_file.read())
            temp_file_path = temp_file.name
        
        # Load CSV data
        df = pd.read_csv(temp_file_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)

        # Convert user input to datetime
        try:
            date_input = pd.to_datetime(date_input)
        except ValueError:
            st.error("Invalid date format. Please enter a valid date (YYYY-MM-DD).")
            os.remove(temp_file_path)
            st.stop()
        
        # Check if 'Close' column exists
        if 'Close' not in df.columns:
            st.error("Missing 'Close' column in CSV file.")
            os.remove(temp_file_path)
            st.stop()

        # Check if entered date exists in dataset
        if date_input in df.index:
            # Select data for the next 7 days
            start_date = date_input
            end_date = start_date + pd.Timedelta(days=6)
            df_subset = df.loc[start_date:end_date]

            # Plot stock prices
            st.write("### 📈 Stock Price Trend")
            fig, ax = plt.subplots()
            ax.plot(df_subset.index, df_subset['Close'], marker='o', linestyle='-')
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price (Close)")
            ax.set_title("7-Day Stock Price Trend")
            st.pyplot(fig)
        else:
            st.error("Date not found in dataset. Please enter a valid date from the uploaded stock data.")
        
        # Clean up temporary file
        os.remove(temp_file_path)
