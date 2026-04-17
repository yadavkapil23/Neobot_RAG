from fastapi import APIRouter, HTTPException

router = APIRouter()

import traceback
from rag import get_smart_rag_response

@router.get("/query/")
async def query_rag_system(query: str, session_id: str = "default_user"):
    try:
        response = await get_smart_rag_response(query, session_id=session_id)
        return {"query": query, "response": response, "session_id": session_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
