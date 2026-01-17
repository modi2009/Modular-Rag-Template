from fastapi import APIRouter

base_router = APIRouter(
    prefix="",
    tags=["base"],
)



@base_router.get("/")
async def read_root():
    return {"message": "Welcome to the Base Route!"}