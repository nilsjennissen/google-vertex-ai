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

- `app.py`: This is the main application file where the Streamlit application is run.
- `scraper.py`: This file contains the code for scraping Google Scholar.
- `vertex_ai.py`: This file contains the code for integrating with Google Vertex AI.
- `langchain.py`: This file contains the code for integrating with Langchain.

## 🗄️ Data

The data used in this project is scraped from Google Scholar based on the user's search specifications. The resulting dataframe of papers and the PDF papers are stored locally, sorted by dates, and the papers are renamed with their title.

## 📚 References

- [Google Vertex AI](https://cloud.google.com/vertex-ai)
- [Langchain](https://www.langchain.ai/)
- [Streamlit](https://streamlit.io/)

## 🏆 Conclusion

This project provides a powerful tool for researchers, enabling them to easily search for and download academic papers from Google Scholar. The integration with Google Vertex AI and Langchain further enhances its capabilities, allowing users to create a knowledge base from the downloaded papers and answer questions regarding the content.

## 🤝 Contributions

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your_username/your_project/issues). Please adhere to this project's [code of conduct](https://github.com/your_username/your_project/blob/main/CODE_OF_CONDUCT.md).├── credentials.py
├── main.py
├── pages/
│   ├── vertex_agent.py
│   ├── vertex_langchain.py
│   └── vertex_researcher.py
├── papers/
│   └── scholar_results_2023-07-10_13-31-49.csv
├── participant-sa-15-ghc-016.json
├── requirements.txt
└── searches/
    ├── scholar_results_2023-07-10_13-32-56.csv
    ├── scholar_results_2023-07-10_13-36-46.csv
    ├── scholar_results_2023-07-10_13-42-39.csv
    ├── scholar_results_2023-07-10_13-55-26.csv
    ├── scholar_results_2023-07-10_13-56-53.csv
    ├── scholar_results_2023-07-10_13-57-23.csv
    ├── scholar_results_2023-07-10_13-57-42.csv
    ├── scholar_results_2023-07-10_14-02-56.csv
    ├── scholar_results_2023-07-10_14-04-21.csv
    └── scholar_results_2023-07-10_14-06-00.csv