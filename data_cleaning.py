#importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer  # --- NEW: for better missing value handling --- done by gpt
from sklearn.compose import ColumnTransformer  # --- NEW: to build a reusable preprocessor pipeline ---gpt
from sklearn.pipeline import Pipeline
import joblib
#Defining the main class 
class DataCleaner:
    def __init__(self,df:pd.DataFrame): #i've defined the class 
        self.df= df.copy()#to store the copy of dataframe
        self.log=[]#it's an empty list
        self.preprocessor = None # --- NEW: to store fitted preprocessing pipeline for later use ---gpt
 # --- NEW: better missing value handling for both numeric and categorical ---gpt
    def handle_missing_values(self, drop_threshold=0.9):
        #dropping columns which have more than drop_threshold (e.g., 90%) missing values
        cols_to_drop = [col for col in self.df.columns if self.df[col].isnull().mean() > drop_threshold]
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            self.log.append(f"Dropped columns with >{drop_threshold*100:.0f}% missing: {cols_to_drop}")
        #filling missing values for numeric columns with median
        for col in self.df.select_dtypes(include=['int64', 'float64']).columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
                self.log.append(f"Filled {missing_count} missing values in '{col}' with median")
        
        # --- NEW: filling missing values for categorical columns ---gpt
        for col in self.df.select_dtypes(include=['object']).columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                self.df[col].fillna("__MISSING__", inplace=True)
                self.log.append(f"Filled {missing_count} missing values in '{col}' with '__MISSING__' placeholder")
        return self.df

#now i am removing all the duplicates by defining a new fuction for it 
    def remove_duplicates(self):
        before = self.df.shape[0]
        self.df.drop_duplicates(inplace=True)
        after = self.df.shape[0]
        removed = before - after
        self.log.append(f"{removed}is removed")
        return self.df
#removing the data which is too high or too low (outliers) cuz it ruins the average
    # --- NEW: fixed outlier handling to avoid removing too many rows sequentially ---gpt
    def handle_outliers(self, z_thresh=3):
        numeric_col = self.df.select_dtypes(include=[np.number]).columns
        before = self.df.shape[0]
        mask = pd.Series(True, index=self.df.index)  # start with all rows kept
        for col in numeric_col:
            std = self.df[col].std()
            if std == 0 or pd.isnull(std):  # skip constant or NaN std columns
                continue
            Z_score = (self.df[col] - self.df[col].mean()) / std
            mask &= (np.abs(Z_score) < z_thresh) | (Z_score.isnull())
        self.df = self.df[mask]
        after = self.df.shape[0]
        self.log.append(f"Removed {before - after} outliers using Z-score threshold {z_thresh}")
        return self.df
#It converts textual (categorical) columns into numbers, which are required for ML models or numeric analysis.
 # --- NEW: replaced manual encoding with scikit-learn transformers for stability ---gpt
    def encode_categoricals(self):
        cat_cols = self.df.select_dtypes(include=['object']).columns
        low_card = [col for col in cat_cols if self.df[col].nunique() <= 10]
        high_card = [col for col in cat_cols if self.df[col].nunique() > 10]

        transformers = []
        if len(low_card) > 0:
            transformers.append(('low_card', OneHotEncoder(handle_unknown='ignore', sparse=False), low_card))
            self.log.append(f"One-hot encoded low-cardinality columns: {low_card}")
        if len(high_card) > 0:
            transformers.append(('high_card', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), high_card))
            self.log.append(f"Ordinal encoded high-cardinality columns: {high_card}")

        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        num_pipeline = Pipeline([
            ('scaler', StandardScaler())  # --- NEW: standardize numeric columns only ---
        ])
        if len(num_cols) > 0:
            transformers.append(('num', num_pipeline, num_cols))
            self.log.append("Standardized all numeric columns (excluding one-hots)")

        self.preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        self.df = pd.DataFrame(
        self.preprocessor.fit_transform(self.df),
        columns=self.preprocessor.get_feature_names_out())
        return self.df
    def save_preprocessor(self, path="preprocessor.joblib"):
        if self.preprocessor:
            joblib.dump(self.preprocessor, path)
            self.log.append(f"Saved preprocessor object to '{path}'")
        else:
            self.log.append("No preprocessor to save")

#now this is final clean data
    def clean_data(self, export_path="cleaned_data.csv", save_preprocessor_path=None):
        self.handle_missing_values()
        self.remove_duplicates()
        self.handle_outliers()
        self.encode_categoricals()
        self.df.to_csv(export_path, index=False)
        self.log.append(f"Exported cleaned dataset to '{export_path}'.")
        if save_preprocessor_path:
            self.save_preprocessor(save_preprocessor_path)        
        return self.df, self.log
