from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_array, check_is_fitted
import joblib

app = FastAPI()

class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier_for_cat1):
        self.classifier = base_classifier_for_cat1
        self.classifiers_ = {}

    def fit(self, X, y_cat1, y_cat2, y_cat3):

        self.classifiers_['Cat1'] = make_pipeline(
            TfidfVectorizer(max_features = 50000, min_df = 2),
            clone(self.classifier)
            ).fit(X, y_cat1)

        for category in y_cat1.unique():
            mask = y_cat1 == category
            if mask.any():
                y_subset = y_cat2[mask]
                if len(y_subset.unique()) > 1:
                    classifier = make_pipeline(
                        TfidfVectorizer(max_features = 50000, min_df = 2),
                        clone(self.classifier)
                    )
                    self.classifiers_['Cat2_' + category] = classifier.fit(X[mask], y_subset)
                else:
                    print(f"Skipping training for Cat2_{category} due to single class: '{y_subset.unique()[0]}'")

        for category in y_cat2.unique():
            mask = y_cat2 == category
            if mask.any():
                y_subset = y_cat3[mask]
                if len(y_subset.unique()) > 1:
                    classifier = make_pipeline(
                        TfidfVectorizer(max_features = 50000, min_df = 2),
                        clone(self.classifier)
                    )
                    self.classifiers_['Cat3_' + category] = classifier.fit(X[mask], y_subset)
                else:
                    print(f"Skipping training for Cat3_{category} due to single class: '{y_subset.unique()[0]}'")

        return self

    def predict(self, X):
        y_pred_cat1 = self.classifiers_['Cat1'].predict(X)

        y_pred_cat2 = []
        y_pred_cat3 = []

        for i, x in enumerate(X):
            cat1 = y_pred_cat1[i]
            cat2_classifier_key = 'Cat2_' + cat1
            if cat2_classifier_key in self.classifiers_:
                cat2 = self.classifiers_[cat2_classifier_key].predict([x])[0]
            else:
                cat2 = ''
            y_pred_cat2.append(cat2)

            cat3_classifier_key = 'Cat3_' + cat2
            if cat3_classifier_key in self.classifiers_:
                cat3 = self.classifiers_[cat3_classifier_key].predict([x])[0]
            else:
                cat3 = ''
            y_pred_cat3.append(cat3)

        return y_pred_cat1, y_pred_cat2, y_pred_cat3


import pandas as pd
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv("app/amazon/train_40k.csv")
val_df = pd.read_csv("app/amazon/val_10k.csv")

X_train = train_df['Text']

y_train_cat1 = train_df['Cat1']
y_train_cat2 = train_df['Cat2']
y_train_cat3 = train_df['Cat3']

X_val = val_df['Text']
y_val_cat1 = val_df['Cat1']
y_val_cat2 = val_df['Cat2']
y_val_cat3 = val_df['Cat3']

hierarchical_clf = HierarchicalClassifier(base_classifier_for_cat1 = LogisticRegression(C = 1e2, n_jobs = 4, solver = 'lbfgs',multi_class='multinomial', random_state = 17,verbose = 0,fit_intercept = True))

hierarchical_clf.fit(X_train, y_train_cat1, y_train_cat2, y_train_cat3)
joblib.dump(hierarchical_clf,'model.joblib')

def predict_text(text):
  y_pred_cat1, y_pred_cat2, y_pred_cat3 = hierarchical_clf.predict([text])
  return y_pred_cat1[0], y_pred_cat2[0], y_pred_cat3[0]

class Comment(BaseModel):
    text: constr(strict=True)

def predict(comment: str) -> str:
    return f"Ваш комментарий: '{comment}' был обработан!"


@app.post("/predict/")
async def get_prediction(comment: Comment):
    if not isinstance(comment.text, str):
        raise HTTPException(status_code = 400, detail="Комментарий должен быть строкой.")

    prediction = predict_text(comment.text)
    return {"prediction": prediction}


