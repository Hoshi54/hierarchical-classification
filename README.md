# hierarchical-classification
hierarchical classification + fastapi + docker

В файле main.ipynb представлены задания с 1-ого по 5-ое и 7-ое

Также в папке app реализовано 6-ое задание

# Задание

Заданием было реализоывать иерархический классификатор

В файле main я реализовал класс HierarchicalClassifier, который внутри себя содержит функции по обучению и предсказанию. На вход класса подается любой sklearn метод, а в качестве метода векторизации данных использовался старенький, но любимый для логистической регрессии TfIdf. Также в этом файле представлено сравнение двух алгоритмов, а именно иерархический и плоский.

Внутри не была реализована языковая модель, так как даже 40000 наблюдений для тюнинга маленькой модели с LoRa у меня займет пару часов(у меня слабый компьютер), поэтому я и люблю RAG.

# Предобработка признаков

Для предобработки я использовал функцию

    def preprop_text(text):

      nums_filtered_text = re.sub(r'[0-9]+', '', text.lower())
      
      punct_filtered_text = ''.join([ch for ch in nums_filtered_text if ch not in string.punctuation])
      
      tokens = nltk.WordPunctTokenizer().tokenize(punct_filtered_text)
      filtr_stop_words_tokens = [pymorphy2.MorphAnalyzer().parse(token)[0].normal_form for token in tokens
                               if token not in set(stopwords.words('english'))]
      
      norm_tokens = [pymorphy2.MorphAnalyzer().parse(token)[0].normal_form for token in filtr_stop_words_tokens]
  
      return f"{' '.join(norm_tokens)}"

Но он не сильно добавил качества для модели + чтоб его загрузить нужно было потратить час(из-за MorphAnalyzer), поэтому я убрал данную функцию из кода

И скажу, что я пробовал обычную лемматизацию с убиранием стоп-слов и пунктуации, но это ухудшило метрикик

# Для активации Docker

$ docker build -t sklearn_fastapi_docker .

$ docker run -p 8000:8000 sklearn_fastapi_docker
