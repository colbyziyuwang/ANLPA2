import streamlit as st
import subprocess  # To run ollama pull
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceWindowNodeParser
import os
import matplotlib.pyplot as plt
import pandas as pd
import re

# ✅ Define Parent Folder for Filings
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/"
STOCK_DATA_FOLDER = os.path.join(PARENT_FOLDER, "stock-data")

# ✅ Function to Extract Filing Date
def extract_filing_date(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    filed_date_match = re.search(r'FILED AS OF DATE:\s*(\d{8})', content)
    if filed_date_match:
        filed_date = filed_date_match.group(1)
        return f"{filed_date[:4]}-{filed_date[4:6]}-{filed_date[6:]}"  # Convert to YYYY-MM-DD
    return None

# ✅ Initialize Streamlit Session State for Persistence
for key in ["filing_date", "file_path", "analyze_stock"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ✅ Streamlit UI
st.title("💸 AI-Powered SEC Filing Detective 🕵️‍♂️")

# ✅ Model Selection
st.write("### 🤖 Choose Your AI Assistant (Free Ollama Models)")
available_models = ["llama3", "mistral", "phi3", "gemma", "qwen2.5", "deepseek-r1"]
bot_choice = st.selectbox("Pick a chatbot model:", available_models)

if st.button("Pull Selected Model"):
    st.write(f"⏳ Pulling `{bot_choice}` model... (This may take a moment if not already downloaded)")
    subprocess.run(["ollama", "pull", bot_choice], check=True)
    st.success(f"✅ `{bot_choice}` model is ready!")

# ✅ Load LLM & Embeddings
llm = Ollama(model=bot_choice, context_window=4096, request_timeout=600.0, temperature=0.1)
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = llm
Settings.embed_model = embedding_model

# ✅ SEC Filing Search
st.write("### 🔍 Search for SEC Filings")

cik = st.text_input("Enter the 10-digit CIK of the company (e.g., 0000320193):")
filing_type = st.selectbox("Select the filing type:", ["10-K", "DEF 14A"])

if st.button("Search Company Filings"):
    st.write(f"🔍 Searching for {filing_type} filings for CIK: {cik}...")
    folder_path = os.path.join(PARENT_FOLDER, filing_type)
    filing_folder = os.path.join(folder_path, cik)
    
    if not os.path.exists(filing_folder) or not os.listdir(filing_folder):
        st.error(f"❌ No {filing_type} filings found for CIK: {cik}")
    else:
        st.success(f"📂 Found {filing_type} filings for CIK: {cik}")
        
        year = st.number_input("Enter a year between 2001 and 2023:", min_value=2001, max_value=2023)
        files = [file for file in os.listdir(filing_folder) if str(year) in file]

        if files:
            found_file = sorted(files)[0]
            st.success(f"📂 Found file: `{found_file}`")

            file_path = os.path.join(filing_folder, found_file)
            extracted_date = extract_filing_date(file_path)

            if extracted_date:
                st.session_state.filing_date = extracted_date  # ✅ Store filing date persistently
                st.session_state.file_path = file_path  # ✅ Store file path persistently
                st.success(f"📅 Filing Date: {st.session_state.filing_date}")
            else:
                st.warning("⚠️ Filing date not found in the document.")

# ✅ Display the Persisted Filing Date
if st.session_state.filing_date:
    st.write(f"📅 **Persisted Filing Date:** {st.session_state.filing_date}")

# ✅ AI Chatbot for SEC Filings
st.write("### 💬 AI Chat with SEC Filing Data")

if st.session_state.filing_date and st.session_state.file_path:
    user_query = st.text_input(
        "Ask a question about the SEC filing:",
        "Give me potential reasons the stock will go up or down. "
        "Give a score between -1 and 1 to indicate decreasing to increasing."
    )

    if st.button("Analyze SEC Filing"):
        docs = SimpleDirectoryReader(input_files=[st.session_state.file_path]).load_data()
        nodes = SentenceWindowNodeParser.from_defaults(
            window_size=1, window_metadata_key="window", original_text_metadata_key="original_text"
        ).get_nodes_from_documents(docs)

        index = VectorStoreIndex(nodes)
        retriever = index.as_retriever(similarity_top_k=3)
        query_engine = RetrieverQueryEngine(retriever=retriever)

        response = query_engine.query(user_query)
        st.subheader("📊 RAG Response:")
        st.write(response.response)

# ✅ Stock Price Analysis
st.write("### 📈 Stock Price Analysis")

if st.session_state.filing_date:
    if cik:
        stock_files = [file for file in os.listdir(STOCK_DATA_FOLDER) if file.startswith(cik)]
        
        if stock_files:
            stock_file_path = os.path.join(STOCK_DATA_FOLDER, sorted(stock_files)[0])  # Pick the first matching file
            st.success(f"📂 Found stock data file: `{os.path.basename(stock_file_path)}`")

            # ✅ Store state to avoid disappearing button
            if st.button("Analyze Stock"):
                st.session_state.analyze_stock = True

            if st.session_state.analyze_stock:
                st.write("We will plot the 7-day price of the stock starting from the filing date.")

                # Load CSV data
                df = pd.read_csv(stock_file_path, parse_dates=['Date'])
                df.set_index('Date', inplace=True)

                try:
                    filing_date = pd.to_datetime(st.session_state.filing_date)  # ✅ Use persisted filing date
                except ValueError:
                    st.error("Unable to convert filing date to datetime.")
                    st.stop()

                if 'Close' not in df.columns:
                    st.error("Missing 'Close' column in CSV file.")
                    st.stop()

                if filing_date in df.index:
                    df_subset = df.loc[filing_date:filing_date + pd.Timedelta(days=6)]
                    st.write("### 📈 Stock Price Trend")
                    fig, ax = plt.subplots()
                    ax.plot(df_subset.index, df_subset['Close'], marker='o', linestyle='-')
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Stock Price (Close)")
                    ax.set_title("7-Day Stock Price Trend")
                    st.pyplot(fig)
                else:
                    st.error("Filing date not found in the dataset.")
        else:
            st.error(f"❌ No stock data file found for CIK: {cik}.")
else:
    st.warning("⚠️ No filing date available. Please search for SEC filings first.")
