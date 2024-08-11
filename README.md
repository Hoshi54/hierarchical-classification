# hierarchical-classification
hierarchical classification + fastapi + docker

В файле main.ipynb представлены задания с 1-ого по 5-ое и 7-ое

Также в папке app реализовано 6-ое задание

# Задание

Заданием было реализоывать иерархический классификатор

В файле main я реализовал класс HierarchicalClassifier, который внутри себя содержит функции по обучению и предсказанию. На вход класса подается любой sklearn метод, а в качестве метода векторизации данных использовался старенький, но любимый для логистической регрессии TfIdf. Также в этом файле представлено сравнение двух алгоритмов, а именно иерархический и плоский.

Внутри не была реализована языковая модель, так как даже 40000 наблюдений для тюнинга маленькой модели с LoRa у меня займет пару часов(у меня слабый компьютер), поэтому я и люблю RAG.

# Для активации Docker

$ docker build -t sklearn_fastapi_docker .
$ docker run -p 8000:8000 sklearn_fastapi_docker
