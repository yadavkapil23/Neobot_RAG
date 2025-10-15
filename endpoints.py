from fastapi import APIRouter, HTTPException

router = APIRouter()

from rag import get_smart_rag_response

@router.get("/query/")
async def query_rag_system(query: str):
    try:
        response = await get_smart_rag_response(query)
        return {"query": query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
