from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import pandas as pd

logger = logging.getLogger("uvicorn.error")

class SimilarItems():
    def __init__(self):
        self._similar_items = None

    def load(self, path, **kwargs):
        logger.info(f"Loading similar items from {path}")
        self._similar_items = pd.read_parquet(path, **kwargs)
        self._similar_items = self._similar_items.set_index('item_id')
        logger.info("Loaded")

    def get(self, item_id: int, k: int = 10):
        try:
            sim_items = self._similar_items.loc[item_id].head(k)
            sim_items = sim_items[['similar_item_id', 'score']].to_dict(orient='list')
        except:
            logger.error("No recommendations for item_id: ", item_id)
            sim_items = {'similar_item_id': [], 'score': []}

        return sim_items

similar_items = SimilarItems()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting")
    similar_items.load('data/similar_items.parquet', columns=['item_id', 'similar_item_id', 'score'])
    yield
    logger.info("Stopping")

app = FastAPI(title='features', lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/similar_items/{item_id}")
async def get_similar_items(item_id: int, k: int = 10):
    """
    Возвращает список похожих объектов длиной k для item_id
    """
    return similar_items.get(item_id, k)