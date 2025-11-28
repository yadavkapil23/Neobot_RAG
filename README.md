---
title: RAG Project
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
python_version: 3.10
---

# 🚀 RAG System with LangChain and FastAPI 🌐

Welcome to this repository! This project demonstrates how to build a powerful RAG system using **LangChain** and **FastAPI** for generating contextually relevant and accurate responses by integrating external data into the generative process.

## 📋 Project Overview

The RAG system combines retrieval and generation to provide smarter AI-driven responses. Using **LangChain** for document handling and embeddings, and **FastAPI** for deploying a fast, scalable API, this project includes:

- 🗂️ **Document Loading**: Load data from various sources (text, PDFs, etc.).
- ✂️ **Text Splitting**: Break large documents into manageable chunks.
- 🧠 **Embeddings**: Generate vector embeddings for efficient search and retrieval.
- 🔍 **Vector Stores**: Store embeddings in a vector store for fast similarity searches.
- 🔧 **Retrieval**: Retrieve the most relevant document chunks based on user queries.
- 💬 **Generative Response**: Use retrieved data with language models (LLMs) to generate accurate, context-aware answers.
- 🌐 **FastAPI**: Deploy the RAG system as a scalable API for easy interaction.

## ⚙️ Setup and Installation

### Prerequisites

Make sure you have the following installed:
- 🐍 Python 3.10+
- 🐳 Docker (optional, for deployment)
- 🛠️ PostgreSQL or FAISS (for vector storage)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yadavkapil23/Neobot_RAG.git
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI server**:
   ```bash
   uvicorn main:app --reload
   ```

   Now, your FastAPI app will be running at `http://127.0.0.1:8000` 🎉!

### Set up Ollama 🦙

This project uses Ollama to run local large language models.

1.  **Install Ollama:** Follow the instructions on the [Ollama website](https://ollama.ai/) to download and install Ollama.

2.  **Pull a model:** Pull a model to use with the application. This project uses `llama3`.
    ```bash
    ollama pull llama3
    ```


## 🛠️ Features

- **Retrieval-Augmented Generation**: Combines the best of both worlds—retrieving relevant data and generating insightful responses.
- **Scalable API**: FastAPI makes it easy to deploy and scale the RAG system.
- **Document Handling**: Supports multiple document types for loading and processing.
- **Vector Embeddings**: Efficient search with FAISS or other vector stores.

## 🛡️ Security

- 🔐 **OAuth2 and API Key** authentication support for secure API access.
- 🔒 **TLS/SSL** for encrypting data in transit.
- 🛡️ **Data encryption** for sensitive document storage.

## 🚀 Deployment

### Hugging Face Spaces (Docker) Deployment
This project is configured for a Hugging Face Space using the Docker runtime.

1. Push this repository to GitHub (or connect local).
2. Create a new Space on Hugging Face → Choose "Docker" SDK.
3. Point it to this repo. Spaces will build using the `Dockerfile` and run `uvicorn` binding to the provided `PORT`.
4. Ensure the file `data/sample.pdf` exists (or replace it) to allow FAISS index creation on startup.

Notes:
- Models `Qwen/Qwen2-0.5B-Instruct` and `all-MiniLM-L6-v2` will be downloaded on first run; initial cold start may take several minutes.
- Dependencies are CPU-friendly; no GPU is required.
- If you see OOM, consider reducing `max_new_tokens` in `vector_rag.py` or swapping to an even smaller instruct model.

### Docker Deployment (Local)
If you want to deploy your RAG system using Docker, simply build the Docker image and run the container:

```bash
docker build -t rag-system .
docker run -p 8000:8000 rag-system
```

### Cloud Deployment
Deploy your RAG system to the cloud using platforms like **AWS**, **Azure**, or **Google Cloud** with minimal setup.

## 🧠 Future Enhancements

- 🔄 **Real-time Data Integration**: Add real-time data sources for dynamic responses.
- 🤖 **Advanced Retrieval Techniques**: Implement deep learning-based retrievers for better query understanding.
- 📊 **Monitoring Tools**: Add monitoring with tools like Prometheus or Grafana for performance insights.

## 🤝 Contributing

Want to contribute? Feel free to fork this repository, submit a pull request, or open an issue. We welcome all contributions! 🛠️

## 📄 License

This project is licensed under the MIT License.

---

🎉 **Thank you for checking out the RAG System with LangChain and FastAPI!** If you have any questions or suggestions, feel free to reach out or open an issue. Let's build something amazing!
