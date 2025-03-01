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
st.title("📈 SEC Filing Analysis RAG Chatbot")
st.write("Upload a 10-K filing and ask: 'Give me potential reasons the stock will go up or down after reading this file.'")

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
