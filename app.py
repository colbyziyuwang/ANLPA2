import streamlit as st
import subprocess  # To run ollama pull
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceWindowNodeParser

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

# ✅ Initialize Streamlit Session State
for key in ["filing_date", "file_path", "analyze_stock", "chatbot_response", "show_content"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "show_content" else False

# ✅ Streamlit UI
st.title("💸 AI-Powered SEC Filing Detective 🕵️‍♂️")
st.write("### ℹ️ About")
st.write("Welcome to the **SEC Filing Analysis RAG Chatbot**, where serious finance meets questionable humor. 🚀")

st.write("This chatbot reads 10-K and DEF 14A filings so you don’t have to. Ask it why a stock might go 🚀 or 📉 after reading financial statements.")

# ✅ Fun Character Picker
st.write("#### 🎭 Pick a Character for Fun")

# ✅ Character options sorted from **high to low** risk
characters = {
    "Nezha": {
        "response": "Nezha: **All in!** 🚀 If it moves, he buys it. High risk, high reward. Doesn't believe in stop-loss. **Risk? He laughs at it.**🔥",
        "risk": 1.0,
    },
    "Elon Musk": {
        "response": "Elon Musk: A **calculated risk-taker**. Loves **high-stakes bets** on AI, rockets, and crypto. Tweets can move markets. 🚀",
        "risk": 0.9,
    },
    "Rick Sanchez": {
        "response": "Rick Sanchez: **Unhinged market chaos**. Would short Tesla while investing in interdimensional crypto. 🤯 **Extreme risk appetite.**",
        "risk": 0.8,
    },
    "Batman": {
        "response": "Batman: A **balanced investor**. Owns companies, reads SEC filings at night, and plays **defensive but strategic**. 🦇",
        "risk": 0.5,
    },
    "Shrek": {
        "response": "Shrek: **A cautious, value investor**. He believes in ‘buying the dip’—but only in swamps, not stocks. Prefers assets he understands. 🧅",
        "risk": 0.4,
    },
    "Spongebob": {
        "response": "Spongebob: A **low-risk** investor—would happily work for free, like an unpaid intern. Prefers safe investments. 🧽",
        "risk": 0.3,
    },
    "Patrick": {
        "response": "Patrick: **Risk? What’s that?** This guy would forget he even invested. Prefers **doing nothing** over risky moves. 💰",
        "risk": 0.2,
    },
    "Tang Seng": {
        "response": "Tang Seng: **Ultra-conservative investor.** Would rather put money in government bonds than stocks. **Risk? No, thank you.** 🏦",
        "risk": 0.0,
    },
}

# ✅ Dropdown menu sorted by risk level
sorted_chars = sorted(characters.items(), key=lambda x: x[1]["risk"], reverse=True)
character_name = st.selectbox("Choose your character:", [name for name, _ in sorted_chars])

# ✅ Display character info
if st.button("Pick a Character"):
    selected = characters[character_name]
    st.write(f"**{character_name}** - Risk Tolerance: `{selected['risk']}`")
    st.write(selected["response"])

    # ✅ Auto-decide based on character selection
    if character_name in ["Nezha", "Elon Musk"]:
        st.session_state.show_content = False
        st.write("I will buy it!!!!!! 🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀")
    
    elif character_name == "Tang Seng":
        st.session_state.show_content = False
        st.write("I will never buy this!!!!!! 💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀")
    
    else:
        st.session_state.show_content = True  # Show full UI for technical analysis

# ✅ Show Remaining Content ONLY if "Technics" is Clicked
if st.session_state.show_content:
    # ✅ Model Selection
    st.write("### 🤖 Choose Your AI Assistant (Free Ollama Models)")
    available_models = ["llama3", "mistral", "phi3", "gemma", "qwen2.5", "deepseek-r1"]
    bot_choice = st.selectbox("Pick a chatbot model:", available_models)

    if st.button("Pull Selected Model"):
        st.write(f"⏳ Pulling `{bot_choice}` model... (This may take a moment if not already downloaded)")
        subprocess.run(["ollama", "pull", bot_choice], check=True)
        st.success(f"✅ `{bot_choice}` model is ready!")

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

    # ✅ Stock Price Analysis
    st.write("### 📈 Stock Price Analysis")
    st.write("We will plot the 7-day price of the stock starting from the filing date.")
