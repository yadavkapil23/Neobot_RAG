from vector_rag import query_vector_store
import wikipedia
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from dotenv import load_dotenv

load_dotenv()
model_name = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
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

wikipedia.set_lang("en")

async def get_smart_rag_response(query: str) -> str:
    print(" Received Query:", query)

    # First: Try Wikipedia
    try:
        summary = wikipedia.summary(query, sentences=5) # Dynamically gets summary
        print("Wikipedia summary found.")
        
        prompt = f"""Use the following Wikipedia information to answer the question as clearly as possible.

Wikipedia Context:
{summary}

Question: {query}
Answer:"""
        result = llm.predict(prompt) 
        answer = result.replace(prompt, "").strip() # Cleanup
        return f"[Wikipedia]\n{answer}"
    except wikipedia.exceptions.PageError:
        print("Wikipedia page not found.") # Corrected simple handling
    except wikipedia.exceptions.DisambiguationError as e:
        return f"The query is ambiguous. Did you mean: {', '.join(e.options[:5])}?"

    # Second: Fallback to LLM (no context)
    try:
        print("Fallback: LLM with no context")
        # FALLBACK PROMPT LOGIC RESTORED
        fallback_prompt = f"You are a knowledgeable assistant. Please answer the following question clearly:\n\n{query}"
        llm_answer = llm.predict(fallback_prompt) 
        answer = llm_answer.replace(fallback_prompt, "").strip() # Cleanup
        if answer and "not sure" not in answer.lower():
            return f"[LLM Fallback]\n{answer.strip()}"
    except Exception as e:
        print("Error during LLM fallback:", e)

    #Finally: Fallback to Local Documents
    try:
        print("Fallback: Local vector search")
        vector_answer = query_vector_store(query)
        if vector_answer:
            return f"[Local Document]\n{vector_answer}"
    except Exception as e:
        print("Error during local vector search:", e)

    return "Sorry, I couldn’t find any information to answer your question."