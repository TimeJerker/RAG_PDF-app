import requests
import json

# Настройки
url = "http://localhost:6333/collections/docs/points/scroll"
payload = {
    "limit": 5,
    "with_payload": True,
    "with_vectors": False
}

print(f"🔍 Отправляем запрос к {url}")
print(f"📦 Payload: {json.dumps(payload, indent=2)}")

try:
    # Отправляем запрос
    response = requests.post(url, json=payload)
    
    print(f"\n📊 Статус: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Ответ от Qdrant:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        # Анализируем результат
        if 'result' in data and 'points' in data['result']:
            points = data['result']['points']
            print(f"\n📌 Найдено точек: {len(points)}")
            
            if len(points) > 0:
                for i, point in enumerate(points):
                    print(f"\n--- Точка {i+1} ---")
                    print(f"ID: {point['id']}")
                    if 'payload' in point:
                        print(f"Текст: {point['payload'].get('text', 'НЕТ ТЕКСТА')[:200]}")
                        print(f"Источник: {point['payload'].get('source', 'unknown')}")
            else:
                print("❌ В коллекции нет точек!")
        else:
            print("❌ Странный ответ от Qdrant")
    else:
        print(f"❌ Ошибка: {response.text}")
        
except Exception as e:
    print(f"❌ Исключение: {e}")