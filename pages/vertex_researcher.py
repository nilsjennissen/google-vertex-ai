'''
This Vertex AI Agent is a combination of the Google Vertex AI and the Langchain Agent for research purposes.
'''

#%% ---------------------------------------------  IMPORTS  ----------------------------------------------------------#
import pandas as pd
from datetime import datetime
import streamlit as st
import requests

from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
import vertexai
from vertexai.preview.language_models import ChatModel
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
import tempfile
import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from main import rec_streamlit, speak_answer, get_transcript_whisper
import time
import spacy_streamlit
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from pptx import Presentation
from pptx.util import Inches
from serpapi import GoogleSearch


from langchain.llms import OpenAI
from serpapi import GoogleSearch
from urllib.parse import urlsplit, parse_qsl

from credentials import serp_api_key


current_directory = os.path.dirname(os.path.abspath(__file__))

#%% ----------------------------------------  VERTEXAI PRELOADS -----------------------------------------------------#
# Initialise the vertexai environment
vertexai.init(project="ghc-016", location="us-central1")
embeddings = VertexAIEmbeddings()

# Initialise the vertexai environment
vertexai.init(project="ghc-016", location="us-central1")

# Initialise the chat model
model = ChatModel.from_pretrained("chat-bison@001")
chat = model.start_chat(examples=[])

# --------------------  SETTINGS  -------------------- #
st.set_page_config(page_title="Home", layout="wide")
st.markdown("""<style>.reportview-container .main .block-container {max-width: 95%;}</style>""", unsafe_allow_html=True)

# --------------------- HOME PAGE -------------------- #
st.title("VERTEX AI Research Assistant")
st.write("""Use the power of LLMs with GOOGLE VERTEX AI and LangChain to scan through your documents. Find information 
from google scholar and insight's with lightning speed. 🚀 Create new content with the support of state of the art 
language models and and voice command your way through your documents. 🎙️""")


#%% ---------------------------------------  PREPROCESS DOCUMENTS ----------------------------------------------------#
def organic_results(search_word, start_year, end_year, num_pages):
    print("extracting organic results..")

    params = {
        "api_key": serp_api_key,          # https://serpapi.com/manage-api-key
        "engine": "google_scholar",
        "q": search_word,  # search query
        "hl": "en",        # language
        "as_ylo": start_year,  # from start_year
        "as_yhi": end_year,  # to end_year
        "start": "0"       # first page
    }

    search = GoogleSearch(params)

    organic_results_data = []

    page_count = 0

    while page_count < num_pages:
        results = search.get_dict()

        print(f"Currently extracting page №{results['serpapi_pagination']['current']}..")

        for result in results["organic_results"]:
            position = result["position"]
            title = result["title"]
            publication_info_summary = result["publication_info"]["summary"]
            result_id = result["result_id"]
            link = result.get("link")
            result_type = result.get("type")
            snippet = result.get("snippet")

            try:
              file_title = result["resources"][0]["title"]
            except: file_title = None

            try:
              file_link = result["resources"][0]["link"]
            except: file_link = None

            try:
              file_format = result["resources"][0]["file_format"]
            except: file_format = None

            try:
              cited_by_count = int(result["inline_links"]["cited_by"]["total"])
            except: cited_by_count = None

            cited_by_id = result.get("inline_links", {}).get("cited_by", {}).get("cites_id", {})
            cited_by_link = result.get("inline_links", {}).get("cited_by", {}).get("link", {})

            try:
              total_versions = int(result["inline_links"]["versions"]["total"])
            except: total_versions = None

            all_versions_link = result.get("inline_links", {}).get("versions", {}).get("link", {})
            all_versions_id = result.get("inline_links", {}).get("versions", {}).get("cluster_id", {})

            organic_results_data.append({
              "page_number": results["serpapi_pagination"]["current"],
              "position": position + 1,
              "result_type": result_type,
              "title": title,
              "link": link,
              "result_id": result_id,
              "publication_info_summary": publication_info_summary,
              "snippet": snippet,
              "cited_by_count": cited_by_count,
              "cited_by_link": cited_by_link,
              "cited_by_id": cited_by_id,
              "total_versions": total_versions,
              "all_versions_link": all_versions_link,
              "all_versions_id": all_versions_id,
              "file_format": file_format,
              "file_title": file_title,
              "file_link": file_link,
            })

        if "next" in results.get("serpapi_pagination", {}):
            search.params_dict.update(dict(parse_qsl(urlsplit(results["serpapi_pagination"]["next"]).query)))
        else:
            break

        page_count += 1

    df = pd.DataFrame(organic_results_data)

    # get current date and time
    now = datetime.now()

    # format as string
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    # save to csv
    df.to_csv(f'papers/scholar_results_{dt_string}.csv', index=False)

    return df

#%% ---------------------------------------  PREPROCESS DOCUMENTS ----------------------------------------------------#
def download_file(url):
    # Send a HTTP request to the URL of the file, stream = True means that the file's content will be streamed when accessing the content attribute
    response = requests.get(url, stream=True)

    # Get the file name by splitting the URL at '/' and taking the last element
    file_name = url.split('/')[-1]

    # Add the .pdf extension to the file name
    file_name = f"{file_name}.pdf"

    st.write(f"Downloading file {file_name}..")

    # Open the file and write the content into the file
    with open(os.path.join('papers/', file_name), 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)


#%% ---------------------------------------  PREPROCESS DOCUMENTS ----------------------------------------------------#
# Check if 'selected_papers' and 'selected_papers_urls' are not in the session state
if 'selected_papers' not in st.session_state:
    st.session_state.selected_papers = []
if 'selected_papers_urls' not in st.session_state:
    st.session_state.selected_papers_urls = []

with st.form(key='search_form'):
    # Settings for Google Scholar search
    start_year = st.number_input("Enter start year", 2000, 2023, 2022)
    end_year = st.number_input("Enter end year", 2000, 2023, 2023)
    num_pages = st.number_input("Enter number of pages - caution: many pages require more calls from serpapi", 1, 10, 1)
    search_words = st.text_input("Enter google scholar search words", "artificial intelligence")

    search_button = st.form_submit_button(label='Search for a topic')

if search_button:
    df = organic_results(search_words, start_year, end_year, num_pages)
    # Save the DataFrame in the session state
    st.session_state.df = df
    st.dataframe(df)


with st.form(key='select_form'):
    # Check if 'df' exists in the session state
    if 'df' in st.session_state:
        # Use the DataFrame from the session state
        df = st.session_state.df

        # Filter the DataFrame to only include rows with PDF links
        df_with_pdfs = df[df['file_link'].notna()]

        # Store the selected papers and their URLs in the session state
        selected_papers = st.multiselect('Select papers', df_with_pdfs['title'].tolist(), key='selected_papers')
        st.session_state.selected_papers_urls = [df_with_pdfs[df_with_pdfs['title'] == paper]['file_link'].values[0] for
                                                 paper in selected_papers]

    select_button = st.form_submit_button(label='Select the files')

# Update the session state with the selected papers
if select_button:
    st.write('Selected papers:', selected_papers)

download_button = st.button(label='Download papers')

# Check if the download button is pressed
if download_button:
    # Check if 'selected_papers' and 'selected_papers_urls' exist in the session state
    if 'selected_papers' in st.session_state and 'selected_papers_urls' in st.session_state:
        # Download the selected papers
        for paper, url in zip(st.session_state.selected_papers, st.session_state.selected_papers_urls):
            download_file(url)


# ----------------- SIDE BAR SETTINGS ---------------- #
st.sidebar.subheader("Settings:")
tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", value=False)

# ------------------ FILE UPLOADER ------------------- #
st.sidebar.subheader("File Uploader:")
uploaded_files = st.sidebar.file_uploader("Choose files", type=["csv", "html", "css", "py", "pdf", "ipynb"],
                                          accept_multiple_files=True)
st.sidebar.metric("Number of files uploaded", len(uploaded_files))
st.sidebar.color_picker("Pick a color for the answer space", "#C14531")

# Initialize docsearch as None
docsearch = None

# --------------------- USER INPUT --------------------- #
user_input = st.text_area("")
# If record button is pressed, rec_streamlit records and the output is saved
audio_bytes = rec_streamlit()


# ------------------- TRANSCRIPTION -------------------- #
if audio_bytes or user_input:

    if audio_bytes:
        try:
            with open("audio.wav", "wb") as file:
                file.write(audio_bytes)
        except Exception as e:
            st.write("Error recording audio:", e)
        transcript = get_transcript_whisper("audio.wav")
    else:
        transcript = user_input

    st.write("**Recognized:**")
    st.write(transcript)

    if any(word in transcript for word in ["abort recording"]):
        st.write("... Script stopped by user")
        exit()

    # ----------------------- ANSWER ----------------------- #
    with st.spinner("Fetching answer ..."):
        time.sleep(6)

        llm = VertexAI()

        # Text Splitter
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        #chunks = text_splitter.split_text(text)
        #knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Use the PDF agent to answer the question
        #docs = knowledge_base.similarity_search(transcript)
        # Show the amount of chunks found
        #st.write(f"Found {len(docs)} chunks.")

        #chain = load_qa_chain(llm, chain_type="stuff")
        #answer = chain.run(input_documents=docs, question=transcript)
        #st.write("**AI Response:**", answer)
        #speak_answer(answer, tts_enabled)
        #st.success("**Interaction finished**")

