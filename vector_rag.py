from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    try:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            print("No local documents matched the query.")
            return None

        context = "\n\n".join([getattr(doc, 'page_content', '') for doc in docs if getattr(doc, 'page_content', None)])
        if not context or len(context.strip()) < 20:
            print("Local documents returned no usable context.")
            return None

        prompt = f"Using the context below, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        raw_output = llm.invoke(prompt)
        answer = raw_output.replace(prompt, "").strip()
        if not answer or len(answer) < 20:
            print("Local document answer was too short or empty.")
            return None

        return answer
    except Exception as e:
        print(f"Error in vector search logic: {e}")
    return None
