import os
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    classification_report
)


class AutoML:

    def __init__(self, file_path, target_col, onehot_threshold=10):
        self.path = file_path
        self.target = target_col
        self.onehot_threshold = onehot_threshold   # <= threshold -> OneHot, > threshold -> Ordinal

        self.load_data()
        self.remove_duplicates()
        self.split_data()
        self.clean_data()
        self.detect_task()
        self.handle_missing()
        self.handle_outliers()
        self.feature_selection()
        self.encoding()
        self.encode_target()
        self.train_test_split_data()
        self.train_models()
        self.save_model()

    
    def load_data(self):                     # load dataset

        self.df = pd.read_csv(self.path)
        print("Dataset",self.df.head())
        print("\nShape:", self.df.shape)

    def remove_duplicates(self):             # Remove outliers
         self.df = self.df.drop_duplicates()
        
    def split_data(self):
        self.x = self.df.drop(columns=[self.target])
        self.y = self.df[self.target]

    def clean_data(self):
        for col in self.x.columns:
            if self.x[col].dtype == "object":
                s = (self.x[col]
                    .astype("string")
                    .str.replace(",", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.strip())

                number = s.str.extract(
                    r"([-+]?\d*\.?\d+)",
                    expand=False)

                numeric = pd.to_numeric(
                    number,
                    errors="coerce")

                # Agar 80% values numeric hain
                if numeric.notna().mean() >= 0.80:
                    self.x[col] = numeric

        target = (self.y
                .astype("string")
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip())

        target_numeric = pd.to_numeric(
                        target,
                        errors="coerce")

        # Agar target mostly numeric hai
        if target_numeric.notna().mean() >= 0.80:
            self.y = target_numeric
        else:
            self.y = target

    def detect_task(self):
        if pd.api.types.is_numeric_dtype(self.y):
            # Few unique numeric values
            if self.y.nunique() <= 20:
                self.task = "classification"
            else:
                self.task = "regression"
        else:
            self.task = "classification"
        print("\nTask:", self.task)

    def handle_missing(self):
        drop_cols = []
        for col in self.x.columns:
            missing = (self.x[col].isna().mean() * 100)
            # More than 50% missing
            if missing > 50:
                drop_cols.append(col)
            else:
                # Numerical → Median
                if pd.api.types.is_numeric_dtype(self.x[col]):
                    self.x[col] = self.x[col].fillna(self.x[col].median())

                else:
                    mode = self.x[col].mode()
                    if len(mode) > 0:
                        self.x[col] = self.x[col].fillna(mode[0])

        # Drop columns
        self.x = self.x.drop(columns=drop_cols)

        if self.y.isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(
                self.y ):
                self.y = self.y.fillna(
                    self.y.median())
            else:
                self.y = self.y.fillna(
                    self.y.mode()[0])
        print("\nMissing values handled")


    def handle_outliers(self):
        numeric_cols = self.x.select_dtypes(include="number").columns

        for col in numeric_cols:
            Q1 = self.x[col].quantile(0.25)
            Q3 = self.x[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            self.x[col] = self.x[col].clip(lower,upper)

        print("\nOutliers handled")

    def feature_selection(self):     # feature selection
        numeric_cols = self.x.select_dtypes(include="number").columns
        categorical_cols = self.x.select_dtypes(exclude="number").columns
        important_features = []

        if len(numeric_cols) > 0:
            if pd.api.types.is_numeric_dtype(self.y):

                corr = self.x[numeric_cols].corrwith(self.y)
                important_numeric = corr[corr.abs() >= 0.10].index.tolist()

                important_features.extend(important_numeric)

        for col in categorical_cols:
            if self.x[col].nunique() <= 15:
                important_features.append(col)

        if len(important_features) == 0:
            important_features = (self.x.columns.tolist())

        self.x = self.x[important_features]
        print("\nImportant features:")
        print(important_features)

    def encoding(self):
     
        categorical_cols = self.x.select_dtypes(exclude="number").columns

        if len(categorical_cols) == 0:
            return

        onehot_cols = [
            col for col in categorical_cols
            if self.x[col].nunique() <= self.onehot_threshold
        ]
        ordinal_cols = [
            col for col in categorical_cols
            if col not in onehot_cols
        ]

        # ---- OneHotEncoding (low cardinality) ----
        if len(onehot_cols) > 0:
            self.onehot_encoder = OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore"
            )
            onehot_array = self.onehot_encoder.fit_transform(self.x[onehot_cols])
            onehot_df = pd.DataFrame(
                onehot_array,
                columns=self.onehot_encoder.get_feature_names_out(onehot_cols),
                index=self.x.index
            )
        else:
            self.onehot_encoder = None
            onehot_df = pd.DataFrame(index=self.x.index)

        # ---- OrdinalEncoding (high cardinality) ----
        if len(ordinal_cols) > 0:
            self.ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            ordinal_array = self.ordinal_encoder.fit_transform(self.x[ordinal_cols])
            ordinal_df = pd.DataFrame(
                ordinal_array,
                columns=ordinal_cols,
                index=self.x.index
            )
        else:
            self.ordinal_encoder = None
            ordinal_df = pd.DataFrame(index=self.x.index)

        numeric_df = self.x.drop(columns=categorical_cols)

        self.x = pd.concat([numeric_df, onehot_df, ordinal_df], axis=1)

        print("\nOneHot encoded columns:", onehot_cols)
        print("Ordinal encoded columns:", ordinal_cols)

    def encode_target(self):
      
        if self.task == "classification":
            if not pd.api.types.is_numeric_dtype(self.y):

                self.target_encoder = OrdinalEncoder()
                y_reshaped = self.y.astype(str).values.reshape(-1, 1)
                self.y = self.target_encoder.fit_transform(y_reshaped).ravel()

        print("\nTarget encoding completed")

    def train_test_split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=0.20,
            random_state=42
        )
                                                                            

    def train_models(self):
        if self.task == "regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42)
            }

        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(random_state=42)
            }

        scores = {}
        predictions = {}

        for name, model in models.items():
            model.fit(
                self.x_train,
                self.y_train)

            prediction = model.predict(self.x_test)
            predictions[name] = prediction

            # Regression
            if self.task == "regression":
                score = r2_score(self.y_test,prediction)
            # Classification
            else: score = accuracy_score(self.y_test,prediction)
            scores[name] = score

        for name, score in scores.items():
            print(name,"=>",round(score, 4))

     # Best Model
        self.best_model_name = max(scores,key=scores.get)
        self.model = models[self.best_model_name]

        self.y_pred = predictions[self.best_model_name] 
        self.scores = scores

        print( "\nBest Model:",
            self.best_model_name)

        print("Best Score:",
            round(scores[self.best_model_name],4))

        # Classification report
        if self.task == "classification":
            print("\nClassification Report:")
            print(
                classification_report(self.y_test,self.y_pred))

    def save_model(self):     # Save model
        data = {
            "model": self.model,
            "task": self.task,
            "columns":self.x.columns.tolist()}

        dataset_name = os.path.splitext(os.path.basename(self.path))[0]
        model_filename = f"{dataset_name}.pkl"

        with open(model_filename,"wb" ) as file:
            pickle.dump(data,file)
        print(f"\nModel saved as {model_filename}")


#  call
automl = AutoML(
    "Attrition.csv",
    "Attrition"
)