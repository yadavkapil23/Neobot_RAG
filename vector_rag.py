from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings 
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

if not chunks:
    raise ValueError("No document chunks found. Ensure 'sample.pdf' exists and is readable.")

# Embed & store (HuggingFace Embeddings are free and fast)
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large") 
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()

# 1. NEW MODEL NAME
model_name = "Qwen/Qwen2-1.5B-Instruct" 

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

# 2. Use the pipeline for text generation
llm_pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=512, 
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

def query_vector_store(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if docs:
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"""
        
        raw_output = llm.predict(prompt)
        answer = raw_output.replace(prompt, "").strip()
        return answer
    return None