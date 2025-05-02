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

    return message

@router.get(
    '/make-predictions-jack',
    description='Using model for predictions for jacks model'
)
async def model_predicitions_jack(
    question: Annotated[str, Query(min_length=10, max_length=500)]
):
    response = model_predictions.unpickle_jack_model_prediction(question)
    message = {
        "message": response,
        "status": status.HTTP_200_OK
    }
    print(message)
    return message


# @router.get(
#     '/predict-rnn',
#     description='Make disease prediction using RNN model (BiLSTM)'
# )
# async def model_predictions_rnn(
#     question: Annotated[str, Query(min_length=10, max_length=500)]
# ):
#     response = model_predictions.unpickle_model_make_prediciton(question)
#     message = {
#         "model": "rnn",
#         "prediction": response,
#         "status": status.HTTP_200_OK
#     }
#     return message
