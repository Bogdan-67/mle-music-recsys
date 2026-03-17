import requests
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('test_service.log', mode='w')
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

recs_service_url = "http://localhost:8000"
events_service_url = "http://localhost:8020"

def test_service():
    logger.info("=== Старт теста сервиса рекомендаций ===")
    
    # 1. Пользователь без персональных рекомендаций (холодный старт)
    user_id_cold = 999999999
    logger.info(f"Тест 1: Пользователь без персональных рекомендаций, user_id={user_id_cold}")
    resp = requests.get(f"{recs_service_url}/recommendations", params={'user_id': user_id_cold})
    if resp.status_code == 200:
        logger.info(f"Рекомендации: {resp.json()}")
    else:
        logger.error(f"Ошибка {resp.status_code} - {resp.text}")

    # 2. Пользователь с персональными рекомендациями, но без онлайн-истории
    user_id_personal = 398979 # Пользователь с наибольшим количеством прослушиваний
    logger.info(f"Тест 2: Пользователь с персональными рекомендациями, без онлайн-истории, user_id={user_id_personal}")
    resp = requests.get(f"{recs_service_url}/recommendations", params={'user_id': user_id_personal})
    if resp.status_code == 200:
        logger.info(f"Рекомендации: {resp.json()}")
    else:
        logger.error(f"Ошибка {resp.status_code} - {resp.text}")

    # 3. Пользователь с персональными рекомендациями и онлайн-историей
    user_id_mixed = 5 # Среднестатистический пользователь
    logger.info(f"Тест 3: Пользователь с персональными рекомендациями и онлайн-историей, user_id={user_id_mixed}")

    events = ['672706', '58723120']
    for event in events:
        logger.info(f"Отправка онлайн-события: item_id={event}")
        requests.post(f"{events_service_url}/events", params={'user_id': user_id_mixed, 'item_id': event})

    resp = requests.get(f"{recs_service_url}/recommendations", params={'user_id': user_id_mixed})
    if resp.status_code == 200:
        logger.info(f"Рекомендации: {resp.json()}")
    else:
        logger.error(f"Ошибка {resp.status_code} - {resp.text}")
        
    logger.info("=== Тест завершен ===")

if __name__ == "__main__":
    test_service()
