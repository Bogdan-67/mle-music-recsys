from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd
import logging
import requests

logger = logging.getLogger("uvicorn.error")

features_service_url = "http://localhost:8010"
events_service_url = "http://localhost:8020"

class Recommendations():
    def __init__(self):
        self._recs = {'personal': None, "default": None}
        self._stats = {
            'request_personal_count': 0,
            'request_default_count': 0
        }

    def load(self, rec_type, path, **kwargs):
        """
        Загружает рекомендации из файла
        """

        logger.info(f"Loading recommendations, type: {rec_type}")
        self._recs[rec_type] = pd.read_parquet(path, **kwargs)
        if rec_type == 'personal':
            self._recs[rec_type] = self._recs[rec_type].set_index('user_id')
        logger.info("Loaded")

    def get(self, user_id: int, k: int = 30):
        """
        Возвращает топ-k рекомендаций для пользователя
        """
        try:
            recs = self._recs['personal'].loc[user_id]
            recs = recs['item_id'].to_list()[:k]
            self._stats['request_personal_count'] += 1
        except KeyError:
            recs = self._recs['default']
            recs = recs['item_id'].to_list()[:k]
            self._stats['request_default_count'] += 1
        except Exception as e:
            logger.error(f"Error getting recommendations for user {user_id}: {e}")
            recs = []
        
        return recs

    def stats(self):
        logger.info(f"Stats for recommendations")

        for name, value in self._stats.items():
            logger.info(f"{name:<30}: {value}")

rec_store = Recommendations()

def dedup_ids(ids):
    """
    Дедублицирует список идентификаторов, оставляя только первое вхождение
    """
    seen = set()
    ids = [id for id in ids if not (id in seen or seen.add(id))]

    return ids

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting")
    rec_store.load('personal', 'data/recommendations.parquet')
    rec_store.load('default', 'data/top_popular.parquet')
    yield
    logger.info("Stopping")
    rec_store.stats()

app = FastAPI(title='recommendations', lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/recommendations/offline")
async def get_offline_recommendations(user_id: int, count: int = 10):
    """
    Возвращает список оффлайн рекомендаций длиной k для пользователя user_id
    """    

    recs = rec_store.get(user_id, count)
    return {"recs": recs}

@app.get("/recommendations/online")
async def get_online_recommendations(user_id: int, count: int = 10):
    """
    Возвращает список онлайн рекомендаций длиной k для пользователя user_id
    """    

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    latest_events = requests.get(f"{events_service_url}/events/{user_id}", params={'k': count}, headers=headers).json()

    if len(latest_events) > 0:
        items = []
        scores = []

        for event in latest_events:
            recs = requests.get(f"{features_service_url}/similar_items/{event}", params={'k': count}, headers=headers).json()
            items.extend(recs['similar_item_id'])
            scores.extend(recs['score'])

        combined = list(zip(items, scores))
        combined = sorted(combined, key=lambda x: x[1], reverse=True)
        combined = [item_id for item_id, _ in combined]

        recs = dedup_ids(combined)[:count]
    else:
        recs = []

    return {"recs": recs}

@app.get('/recommendations')
async def get_mixed_recommendations(user_id: int, count: int = 10):
    """
    Возвращает список смешанных рекомендаций длиной k для пользователя user_id
    """    

    online_recs = await get_online_recommendations(user_id, count)
    offline_recs = await get_offline_recommendations(user_id, count)

    online_recs = online_recs['recs']
    offline_recs = offline_recs['recs']

    min_length = min(len(online_recs), len(offline_recs))

    recs = []
    for i in range(min_length):
        recs.append(online_recs[i])
        recs.append(offline_recs[i])

    recs += online_recs[min_length:]
    recs += offline_recs[min_length:]

    recs = dedup_ids(recs)[:count]

    return {"recs": recs}

