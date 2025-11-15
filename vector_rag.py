from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault('HF_HOME', '/tmp/huggingface_cache')
os.environ.setdefault('TRANSFORMERS_CACHE', '/tmp/huggingface_cache/transformers')
os.environ.setdefault('HF_DATASETS_CACHE', '/tmp/huggingface_cache/datasets')

print("Loading Qwen2-0.5B-Instruct...")
model_name = "Qwen/Qwen2-0.5B-Instruct"  

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="cpu", 
    trust_remote_code=True
)

llm_pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=256, 
    do_sample=True,
    temperature=0.5,
    top_p=0.9,
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 

loader = PyPDFLoader("data/sample.pdf") 
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

if not chunks:
    raise ValueError("No document chunks found.")

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

def query_vector_store(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if docs:
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"""
        
        raw_output = llm.invoke(prompt)
        answer = raw_output.replace(prompt, "").strip()
        return answer
    return None
