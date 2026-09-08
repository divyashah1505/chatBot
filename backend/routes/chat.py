from fastapi import APIRouter
from pydantic import BaseModel

from chatBot.chatbot import get_response
from db.connection import conversations_collection
from db.models import create_conversation


router = APIRouter()


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
def chat(request: ChatRequest):

    # Generate chatbot response
    response = get_response(request.message)

    # Create conversation document
    conversation = create_conversation(
        request.message,
        response
    )

    # Save conversation in MongoDB
    conversations_collection.insert_one(conversation)

    return {
        "reply": response
    }