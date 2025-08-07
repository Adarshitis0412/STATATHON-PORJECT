#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
#Defining the main class 
class DataAnalyzer():
    def __init__(self,df: pd.DataFrame, output_dir="analysis_outputs"):#output_dir="analysis_outputs": Ye optional hai — agar mai kuch nahi deta toh graphs iss naam ke folder me save honge.
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)#exit_ok=true means that if folder pehle sei hi banna hua hai toh error mt deena
        self.analysis_log = []#hr step ka log save krne ke liyea
    def compute_statistics(self):
        desc_stats= self.df.describe(include='all').T#it describes the stats and .T -> transposes the row and col
        self.analysis_log.append("Generated descriptive statistics")
        return desc_stats
    def correlation_heatmap(self, save_as="correlation_heatmap.png"):
        numeric_df= self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        heatmap_path =os.path.join(self.output_dir, save_as)
        plt.title("Correlation HeatMap")
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()
        self.analysis_log.append(f"Saved correlation heatmap as '{heatmap_path}'.")
    def cross_tab_analysis(self,col1,col2):
        if col1 not in self.df.columns or col2 not in self.df.columns:#check wether both the columns exixts or not
           return f"Columns '{col1}' or '{col2}' not found."
        ct = pd.crosstab(self.df[col1],self.df[col2])
        self.analysis_log.append(f"Generated cross-tab between '{col1}' and '{col2}'.")
        return ct
    def trend_analysis(self, date_col, target_col, save_as="trend_plot.png"):
        if date_col not in self.df.columns or target_col not in self.df.columns:
         return f"Columns '{date_col}' or '{target_col}' not found."
        self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
        trend_df = self.df[[date_col, target_col]].dropna()
        trend_df = trend_df.groupby(date_col)[target_col].mean()
        plt.figure(figsize=(10, 5))
        trend_df.plot()
        plt.xlabel("Date")
        plt.ylabel(f"Average {target_col}")
        plt.title(f"Trend of {target_col} over time")
        plt.tight_layout()
        trend_path = os.path.join(self.output_dir, save_as)
        plt.savefig(trend_path)
        plt.close()
        self.analysis_log.append(f"Saved trend plot as '{trend_path}'.")
        return trend_path
    def run_full_analysis(self, date_col=None, trend_col=None):
        results = {
        "statistics": self.compute_statistics(),
        "correlation_heatmap": self.correlation_heatmap()
    }

        if date_col and trend_col:
         results["trend_plot"] = self.trend_analysis(date_col, trend_col)

        return results, self.analysis_log
