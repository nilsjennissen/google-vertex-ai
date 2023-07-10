# 🧭 Project Overview 

This project is centered around the innovative Google Vertex AI Palm2 platform. We have developed a research assistant for Google Scholar using a Streamlit application. The application is designed to take user search specifications and scour Google Scholar across multiple pages. The result is a comprehensive dataframe of all scraped papers and a selection of all downloadable PDF documents. The selected papers can then be downloaded and utilized in conjunction with Vertex AI and Langchain. 

## 🚧 Prerequisites

Before you get started with this project, you need to have the following:

- Python 3.6 or later
- Streamlit
- Google Vertex AI Access Credentials
- Langchain

## 🎛 Project Setup

To set up the project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies using pip install -r requirements.txt.
3. Run the Streamlit application using streamlit run app.py.

## 📦 Project Structure

The project is structured as follows:
```bash
├── credentials.py
├── main.py
├── pages/
│   ├── vertex_agent.py
│   ├── vertex_langchain.py
│   └── vertex_researcher.py
├── papers/
│   └── paper_name.pdf
├── participant-sa-xx-ghc-xxx.json
├── requirements.txt
├── README.md
└── searches/
    └── scholar_results_yyyy-mm-dd_hh-mm-ss.csv
````

## 🗄️ Data

The data used in this project is scraped from Google Scholar based on the user's search specifications. The resulting dataframe of papers and the PDF papers are stored locally, sorted by dates, and the papers are renamed with their title.

## 📚 References

- [Google Vertex AI](https://cloud.google.com/vertex-ai)
- [Langchain](https://python.langchain.com/docs/get_started/introduction.html)
- [Streamlit](https://streamlit.io/)

## 🏆 Conclusion

This project provides a powerful tool for researchers, enabling them to easily search for and download academic papers from Google Scholar. The integration with Google Vertex AI and Langchain further enhances its capabilities, allowing users to create a knowledge base from the downloaded papers and answer questions regarding the content.

## 🤝 Contributions

Contributions, issues, and feature requests are welcome! 

# Business plan 👔

## Executive Summary:

Project aims to develop a research assistant using Google Vertex AI Palm2 platform.
Streamlines searching and accessing academic papers from Google Scholar.
Provides comprehensive dataframe of scraped papers and downloadable PDFs.
Integration with Google Vertex AI and Langchain enhances capabilities.
## Problem:

Researchers face challenges in manual searches, limited access to papers, and organizing information.
Existing methods lack sophistication and efficiency, hindering productivity.
## Solution:

User-friendly interface for inputting search specifications and navigating Google Scholar.
Automatic scraping and organization of relevant papers.
Creates comprehensive dataframe and offers downloadable PDFs.
Integration with Google Vertex AI and Langchain enables knowledge extraction.
## Market Size:

Global academic publishing market projected to reach $32.2 billion by 2026.
Millions of researchers and scholars worldwide as target audience.
Demand for efficient literature research tools is high.
## Revenue Stream:

Subscription-based model with advanced features for researchers.
Licensing opportunities for institutional use.
##  Next Steps/Backlogs:

Refine user interface and optimize search algorithms.
Expand integration with additional AI models for knowledge extraction.
Develop mobile versions for the research assistant.