import os
import signal
from dataclasses import dataclass
from fastapi import Depends, HTTPException, status, Response
from litserve import LitServer
import httpx
from urllib.parse import quote
import traceback

from pydantic import BaseModel, ValidationError

from icicle_playgrounds.inference_servers import InferenceServerAPI
from icicle_playgrounds.inference_servers.config import config
from icicle_playgrounds.schemas.patra_model_cards import PatraModelCard



@dataclass
class Model:
    id: str
    name: str
    obj: object

class PatraModel(BaseModel):
    id: str

class ModelInfo(BaseModel):
    id: str
    name: str

class InferenceServer(LitServer):
    def __init__(self, api: InferenceServerAPI, inference_endpoint: str, *args, **kwargs):
        super().__init__(*args, lit_api=api, api_path=inference_endpoint, **kwargs)
        self._api = api
        self._register_apis()

        self._inference_model: ModelInfo | None = None
        self._loaded_models: dict[str, Model] = {}
        
    @property
    def api(self):
        return self._api

    @property
    def inference_model(self):
        return self._inference_model

    @inference_model.setter
    def inference_model(self, model: ModelInfo):
        self._inference_model = model

    @property
    def loaded_models(self):
        return self._loaded_models

    def _register_apis(self):
        self.app.add_api_route(
            path="/shutdown",
            endpoint=self.shutdown,
            methods=["GET"],
            dependencies=[Depends(self.setup_auth())],
            status_code=status.HTTP_200_OK,
        )
        self.app.add_api_route(
            path="/ready",
            endpoint=self.ready,
            methods=["GET"],
            dependencies=[Depends(self.setup_auth())],
            status_code=status.HTTP_200_OK,
        )
        self.app.add_api_route(
            path="/add-model",
            endpoint=self.add_model,
            methods=["POST"],
            dependencies=[Depends(self.setup_auth())],
            status_code=status.HTTP_201_CREATED,
            response_model=ModelInfo,
        )
        self.app.add_api_route(
            path="/hot-swap-model",
            endpoint=self.hot_swap_model,
            methods=["PUT"],
            dependencies=[Depends(self.setup_auth())],
            response_model=ModelInfo,
            status_code=status.HTTP_200_OK,
        )

    def _add_model(self, model_id: str, model_name: str, model: object):
        if model_id in self._loaded_models:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Model with id {model_id} already loaded.")
        self._loaded_models[model_id] = Model(
            id=model_id,
            name=model_name,
            obj=model
        )
        if self.inference_model is None:
            self.inference_model = ModelInfo(
                id=model_id,
                name=model_name
            )
            self.api.model = model

    def _remove_model(self, model_id: str):
        if model_id not in self._loaded_models:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model with id {model_id} not found.")

        if model_id == self.inference_model.id:
            self.inference_model = None
            del self.api.model

        model: Model = self._loaded_models.pop(model_id)
        return model

    async def shutdown(self):
        os.kill(os.getpid(), signal.SIGTERM)
        return Response(
            status_code=status.HTTP_200_OK, content="Server shutting down..."
        )

    async def ready(self):
        if self.loaded_models and self.inference_model:
            return {
                "status": "ready",
                "inference_model": self.inference_model.model_dump(),
            }
        else:
            return {
                "status": "not ready, no models loaded",
            }

    async def add_model(self, patra_model: PatraModel):
        if not config.PATRA_URL:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error: PATRA_URL is not set."
            )

        url = f"{config.PATRA_URL}/models/{quote(patra_model.id)}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url=url)

            response.raise_for_status()
            model_card_data = response.json()["result"]
            try:
                model_card = PatraModelCard(**model_card_data)
            except ValidationError as ve: # Assuming pydantic.ValidationError is imported
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid model card data: {ve.errors()}")


        model = self.api.load_model(model_card.ai_model.location)
        self._add_model(model_card.ai_model.model_id, model_card.ai_model.name, model)
        return self.inference_model

    async def remove_model(self, patra_model: PatraModel):
        return self._remove_model(patra_model.id)

    async def hot_swap_model(self, patra_model: PatraModel):
        if patra_model.id not in self._loaded_models:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model with id {patra_model.id} not found.")
        self.api.model = self._loaded_models[patra_model.id].obj
        return self.inference_model