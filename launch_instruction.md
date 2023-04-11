# Инструкция по запуску

1. Клониурем репозиторий
2. [Устанавливаем docker](https://www.docker.com/products/docker-desktop)
3. Собираем образы и запускаем контейнеры
   ```
   docker-compose up -d --build 
   ```   
4. Смотрим логи рекомендера
   ```
   docker logs recommender-container
   ```  
5. Скачиваем по ссылке архив со всеми необходимыми данными и скалдываем в папку *data*
   ```
   https://drive.google.com/file/d/1PIGHAJaCKNKqgjgIgSnYbMBKX9F6M_kf/view?usp=sharing
   ```  
6. Переходим в папку *sim*
7. Создаем чистый env с python 3.7
8. Устанавливаем зависимости
   ```
   pip install -r requirements.txt
   ``` 
9. Добавляем текущую директорию в $PYTHONPATH
   ```
   export PYTHONPATH=${PYTHONPATH}:.
   ```  
10. Запускаем A/B эксперимент
   ```
   python sim/run.py --episodes 1000 --config config/env.yml multi --processes 4
   ```  