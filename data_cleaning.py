#importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
#Defining the main class 
class DataCleaner:
    def __init__(self,df:pd.DataFrame): #i've defined the class 
        self.df= df.copy()#to store the copy of dataframe
        self.log=[]#it's an empty list
    def handel_missing_values(self):
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0: #this line basically means if a column has more than 0 missing values do something about it [.isnull-> to check null val, .sum ->tocpunt all missing values of the column]
                 if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(),inplace=True)#filling up the columns with median
        return self.df
#now i am removing all the duplicates by defining a new fuction for it 
    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        return self.df
#removing the data which is too high or too low (outliers) cuz it ruins the average
    def handle_outliers(self,z_thresh=3):#The Z-score threshold. Any value above or below ±3 is considered an outlier.
        numeric_col= self.df.select_dtypes(include=[np.number]).columns#selecting only numectic values in the columns
        before = self.df.shape[0]
        for col in numeric_col:#applying the loop 
            Z_score = (self.df[col] - self.df[col].mean())/self.df[col].std()#applying the formula
            self.df = self.df[(np.abs(Z_score) < z_thresh) | (Z_score.isnull())]#np.abs = to convert all the negative values into positive,< z_thresh: Keeps only rows where Z-score is less than threshold,| (z_scores.isnull()): Keeps rows where Z-score is NaN
        after = self.df.shape[0]
        self.log.append(f"Removed {before - after} outliers using Z-score.")
        return self.df
#It converts textual (categorical) columns into numbers, which are required for ML models or numeric analysis.
    def encode_categoricals(self):
        cat= self.df.select_dtypes(include=['object']).columns
        for col in cat:
            if self.df[col].nunique()<=10:# nunique is functions,it is use to select all the uniqe characters 
                dummies = pd.get_dummies(self.df[col],prefix=col)#dummies is to create new columns,his function takes a categorical column and converts it into multiple binary columns (0 or 1)
                self.df = pd.concat([self.df.drop(col, axis=1), dummies], axis=1)
                self.log.append(f"One-hot encoded '{col}'.")
            else: #this encoder is for more than one or two category
                le = LabelEncoder()
                self.df[col]= le.fit_transform(self.df[col])#TRANSFORMING TRHE COLUMN
                self.log.append(f"Label encoded '{col}'.")
        return self.df
#now applying standardization
    def standardization(self):
        scaler = StandardScaler()
        numeric_col= self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_col]=scaler.fit_transform(self.df[numeric_col])
        self.log.append("Standardized all numeric columns.")
        return self.df
#now this is final clean data
    def clean_data(self, export_path="cleaned_data.csv"):
        self.handel_missing_values()
        self.remove_duplicates()
        self.handle_outliers()
        self.encode_categoricals()
        self.standardization()
        self.df.to_csv(export_path, index=False)
        self.log.append(f"Exported cleaned dataset to '{export_path}'.")
        return self.df, self.log