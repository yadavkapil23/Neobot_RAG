from vector_rag import query_vector_store, llm # <--- FIX: Import llm here!
import wikipedia
# REMOVED: All duplicate model/pipeline/tokenizer imports and initialization code

# The 'llm' instance is now imported from vector_rag.py and is ready to use.
wikipedia.set_lang("en")

async def get_smart_rag_response(query: str) -> str:
    print(" Received Query:", query)

    # First: Try Wikipedia
    try:
        summary = wikipedia.summary(query, sentences=5)
        print("Wikipedia summary found.")
        
        prompt = f"""Use the following Wikipedia information to answer the question as clearly as possible.

Wikipedia Context:
{summary}

Question: {query}
Answer:"""
        result = llm.predict(prompt) 
        answer = result.replace(prompt, "").strip()
        return f"[Wikipedia]\n{answer}"
    except wikipedia.exceptions.PageError:
        print("Wikipedia page not found.")
    except wikipedia.exceptions.DisambiguationError as e:
        return f"The query is ambiguous. Did you mean: {', '.join(e.options[:5])}?"

    # Second: Fallback to LLM (no context)
    try:
        print("Fallback: LLM with no context")
        
        fallback_prompt = f"You are a knowledgeable assistant. Please answer the following question clearly:\n\n{query}"
        llm_answer = llm.predict(fallback_prompt) 
        answer = llm_answer.replace(fallback_prompt, "").strip()
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