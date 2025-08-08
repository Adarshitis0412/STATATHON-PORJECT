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
        return heatmap_path
    def missing_value_heatmap(self, save_as="missing_values_heatmap.png"):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        heatmap_path = os.path.join(self.output_dir, save_as)
        plt.title("Missing Values Heatmap")
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()
        self.analysis_log.append(f"Saved missing values heatmap as '{heatmap_path}'.")
        return heatmap_path
    # --- NEW: Top correlation table ---
    def top_correlations(self, target_col, n=5):
        if target_col not in self.df.columns:
            self.analysis_log.append(f"Target column '{target_col}' not found for correlation check")
            return None
        if self.df[target_col].dtype not in [np.float64, np.int64]:
            self.analysis_log.append(f"Target column '{target_col}' is not numeric for correlation check")
            return None
        corr = self.df.corr()[target_col].drop(target_col)
        top_corr = corr.abs().sort_values(ascending=False).head(n)
        self.analysis_log.append(f"Computed top {n} correlations with '{target_col}'.")
        return top_corr
    def cross_tab_analysis(self,col1,col2):
        if col1 not in self.df.columns or col2 not in self.df.columns:#check wether both the columns exixts or not
           return f"Columns '{col1}' or '{col2}' not found."
        ct = pd.crosstab(self.df[col1],self.df[col2])
        self.analysis_log.append(f"Generated cross-tab between '{col1}' and '{col2}'.")
        return ct
    def trend_analysis(self, date_col, target_col, freq=None, rolling_window=None, save_as="trend_plot.png"):
        if date_col not in self.df.columns or target_col not in self.df.columns:
         return f"Columns '{date_col}' or '{target_col}' not found."
        self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
        trend_df = self.df[[date_col, target_col]].dropna()
        trend_df = trend_df.groupby(date_col)[target_col].mean()
        if freq:  # --- NEW: option to resample ---
            trend_df = trend_df.resample(freq).mean()

        if rolling_window:  # --- NEW: rolling mean smoothing ---
            trend_df = trend_df.rolling(window=rolling_window, min_periods=1).mean()

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
    # --- NEW: distribution plots for top N numeric features ---
    def distribution_plots(self, n=5):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        top_cols = numeric_cols[:n]  # just first n numeric cols
        saved_paths = []
        for col in top_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            path = os.path.join(self.output_dir, f"dist_{col}.png")
            plt.savefig(path)
            plt.close()
            saved_paths.append(path)
            self.analysis_log.append(f"Saved distribution plot for '{col}' as '{path}'.")
        return saved_paths
    # --- NEW: full analysis now returns all outputs in a dict ---
    def run_full_analysis(self, date_col=None, trend_col=None, target_col=None):
        results = {
            "statistics": self.compute_statistics(),
            "correlation_heatmap": self.correlation_heatmap(),
            "missing_value_heatmap": self.missing_value_heatmap(),
            "distributions": self.distribution_plots(),
        }
        if target_col:
            results["top_correlations"] = self.top_correlations(target_col)
        if date_col and trend_col:
            results["trend_plot"] = self.trend_analysis(date_col, trend_col)
        return results, self.analysis_log