from fastapi import FastAPI

class EventStore:

    def __init__(self, max_events_per_user=10):

        self.events = {}
        self.max_events_per_user = max_events_per_user

    def put(self, user_id, item_id):
        """
        Сохраняет событие
        """

        user_events = self.events.get(user_id, [])
        self.events[user_id] = [item_id] + user_events[: self.max_events_per_user]

    def get(self, user_id, k):
        """
        Возвращает события для пользователя
        """
        user_events = self.events.get(user_id, [])[:k]

        return user_events

events_store = EventStore()

app = FastAPI(title='events')

@app.post("/events")
async def log_event(user_id: int, item_id: int):
    events_store.put(user_id, item_id)
    return {"message": "Event logged", "result": "ok"}

@app.get("/events/{user_id}")
async def get_user_events(user_id: int, k: int = 10):
    return events_store.get(user_id, k)