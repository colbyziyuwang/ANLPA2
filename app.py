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
for key in ["filing_date", "file_path", "analyze_stock", "chatbot_response"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ✅ Streamlit UI
st.title("💸 AI-Powered SEC Filing Detective 🕵️‍♂️")

# ✅ About
st.write("### ℹ️ About")
st.write("Welcome to the **SEC Filing Analysis RAG Chatbot**, where serious finance meets questionable humor. 🚀")

st.write("This chatbot reads 10-K and DEF 14A filings so you don’t have to. Ask it why a stock might go 🚀 or 📉 after reading financial statements.")

# ✅ Fun Character Picker
st.write("#### 🎭 Pick a Character for Fun")
character_name = st.text_input("Enter your character name:")
if st.button("Pick a Character"):
    character_responses = {
        "spongebob": "Spongebob: Works for free, like an unpaid intern. 🧽",
        "patrick": "Patrick: Financial advice? 'Just don’t spend money.' Genius. 💰",
        "nezha": "Nezha: 我命由我不由天！(Translation: I control my own destiny, not the heavens!)",
        "elon musk": "Elon Musk: Likes rockets, AI, and tweeting at 3 AM. 🚀",
        "batman": "Batman: He doesn’t read SEC filings—he **owns** the companies filing them. 😢",
        "rick sanchez": "Rick Sanchez: '10-K filings? Pfft. Just invest in interdimensional markets, Morty!' 🤯",
        "shrek": "Shrek: 'This chatbot is like an onion—it has layers. Also, I don’t do stocks, I do **swamps**.' 🧅",
        "default": "Hmm... I don't know that character. Maybe they’re off trading crypto?"
    }
    st.write(character_responses.get(character_name.strip().lower(), character_responses["default"]))

# ✅ Initialize Session State to Persist Data
if "show_analysis" not in st.session_state:
    st.session_state.show_analysis = False

# ✅ Button to Toggle All Analysis (Plots + Stats)
if st.button("📊 Show Stock Price Change Analysis & Statistics"):
    st.session_state.show_analysis = not st.session_state.show_analysis

if st.session_state.show_analysis:
    st.write("### 📈 7-Day Stock Price Change Analysis & Statistics")

    # ✅ Display Box Plot
    st.write("#### 📌 Box Plot of 7-Day Stock Price Change")
    box_plot_path = "boxplot-7-day-average-stock-price-change.png"
    st.image(box_plot_path, caption="Box Plot of 7-Day Stock Price Change After Filing", use_container_width=True)

    # ✅ Display Distribution Plot
    st.write("#### 📌 Distribution of 7-Day Stock Price Change")
    dist_plot_path = "distribution-7-day-average-stock-price-change.png"
    st.image(dist_plot_path, caption="Distribution of 7-Day Stock Price Change After Filing", use_container_width=True)

    # ✅ Display Descriptive Statistics
    st.write("### 📊 Descriptive Statistics")
    stats_summary = f"""
    - **Mean:** {0.42:.2f}%
    - **Median:** {0.40:.2f}%
    - **Q1 (25th Percentile):** {-2.16:.2f}%
    - **Q3 (75th Percentile):** {2.98:.2f}%
    - **Interquartile Range (IQR):** {5.14:.2f}%
    - **Standard Deviation:** {5.73:.2f}%
    - **Coefficient of Variation (CV):** {13.68:.2f}
    - **Skewness:** {0.27:.2f} (Indicates asymmetry of distribution)
    - **Kurtosis:** {11.97:.2f} (Indicates tail heaviness)
    """
    st.markdown(stats_summary)

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
    # ✅ Different Queries for 10-K vs. DEF 14A
    if filing_type == "10-K":
        default_query = (
            "Analyze this 10-K filing and identify both positive and negative signals that may impact the stock price. "
            "Consider revenue growth, profitability, debt levels, risk disclosures, new business strategies, and industry trends. "
            "Provide a score between 0 (strong negative impact) and 1 (strong positive impact), ensuring a neutral perspective by weighing both positive and negative aspects. "
            "Explain the key reasons behind the score."
        )
    else:  # DEF 14A (Proxy Statements)
        default_query = (
            "Analyze this DEF 14A proxy statement and determine its potential impact on investor sentiment. "
            "Consider executive compensation, board structure, shareholder proposals, voting outcomes, and corporate governance policies. "
            "Provide a score between 0 (strong negative impact) and 1 (strong positive impact), ensuring a neutral perspective by weighing both positive and negative aspects. "
            "Explain the key reasons behind the score."
        )

    user_query = st.text_input("Ask a question about the SEC filing:", default_query)

    if st.button("Analyze SEC Filing"):
        docs = SimpleDirectoryReader(input_files=[st.session_state.file_path]).load_data()
        nodes = SentenceWindowNodeParser.from_defaults(
            window_size=1, window_metadata_key="window", original_text_metadata_key="original_text"
        ).get_nodes_from_documents(docs)

        index = VectorStoreIndex(nodes)
        retriever = index.as_retriever(similarity_top_k=3)
        query_engine = RetrieverQueryEngine(retriever=retriever)

        response = query_engine.query(user_query)
        st.session_state.chatbot_response = response.response  # ✅ Store chatbot response
        st.subheader("📊 RAG Response:")
        st.write(st.session_state.chatbot_response)

# ✅ Display Previous Chatbot Response (Even After Clicking "Analyze Stock")
if st.session_state.chatbot_response:
    st.subheader("📊 RAG Response (Persisted):")
    st.write(st.session_state.chatbot_response)

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