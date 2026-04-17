from vector_rag import query_vector_store, llm 
import wikipedia
from wikipedia import exceptions as wikipedia_exceptions
from langchain_community.chat_message_histories import ChatMessageHistory

wikipedia.set_lang("en")

session_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

async def get_smart_rag_response(query: str, session_id: str = "default_user") -> str:
    print(f"--- Processing Query: {query} (Session: {session_id}) ---")
    
    history = get_session_history(session_id)
    
    past_messages = history.messages[-6:]
    history_str = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in past_messages])

    final_answer = ""
    source_label = ""

    try:
        vector_answer = query_vector_store(query)
        if vector_answer and len(vector_answer.strip()) > 5:
            final_answer = vector_answer.strip()
            source_label = "[Local Document]"
    except Exception as e:
        print(f"Local search failed: {e}")

    if not final_answer:
        try:
            search_results = wikipedia.search(query, results=5)
            summary = None
            for title in search_results:
                try:
                    summary = wikipedia.summary(title, sentences=3)
                    break
                except (wikipedia_exceptions.DisambiguationError, wikipedia_exceptions.PageError):
                    continue

            if summary:
                prompt = f"Previous Conversation:\n{history_str}\n\nContext:\n{summary}\n\nQuestion: {query}\nAnswer:"
                raw_output = llm.invoke(prompt)
                final_answer = raw_output.replace(prompt, "").strip()
                source_label = "[Wikipedia]"
            else:
                print("Wikipedia lookup found no usable summary.")
        except Exception as e:
            print(f"Wikipedia lookup failed: {e}")

    # 3. Direct LLM Fallback (with memory)
    if not final_answer:
        try:
            prompt = f"Previous Conversation:\n{history_str}\n\nQuestion: {query}\nAnswer:"
            raw_output = llm.invoke(prompt)
            final_answer = raw_output.replace(prompt, "").strip()
            source_label = "[AI Assistant]"
        except Exception as e:
            final_answer = "I'm sorry, I encountered an error."
            source_label = "[Error]"

    # If everything fails
    if not final_answer:
        final_answer = "I couldn't find a detailed answer, but I'm here to chat!"
        source_label = "[General]"

    # Update the memory history for next time
    history.add_user_message(query)
    history.add_ai_message(final_answer)

    return f"{source_label}\n{final_answer}"
