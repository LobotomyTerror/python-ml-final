from typing import Annotated
from fastapi import APIRouter, status, Query

from utils import model_predictions

router = APIRouter()

@router.get(
    '/test',
    description="Simple route"
)
async def test_route():
    return {"message": "Hello!"}


@router.get(
    '/make-predictions',
    description='Using model for predictions'
)
async def model_predicitions(
    question: Annotated[str, Query(min_length=10, max_length=500)]
):
    response = model_predictions.unpickle_model_make_prediciton(question)
    message = {
        "message": response,
        "status": status.HTTP_200_OK
    }
    print(message)
    return message
