# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:43:09 2022

@author: HSelato
"""
import numpy as np
import pandas as pd

class StrategyPerformanceIndex:
    
    def CAGR(self, DF):
        "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
        df = DF.copy()
        df["return"] = DF["CLOSE"].pct_change()
        df["cum_return"] = (1 + df["return"]).cumprod()
        n = len(df)/252
        CAGR = (df["cum_return"])**(1/n) - 1
        return CAGR
        
    def volatility(self, DF):
        "function to calculate annualized volatility of a trading strategy"
        df = DF.copy()
        df["return"] = df["CLOSE"].pct_change()
        vol = df["return"].std() * np.sqrt(252)
        return vol
    
    def sharpe(self, DF, rf):
        "function to calculate Sharpe Ratio of a trading strategy"
        df = DF.copy()
        return (self.CAGR(df) - rf)/self.volatility(df)
    
    def sortino(self, DF, rf):
        "function to calculate Sortino Ratio of a trading strategy"
        df = DF.copy()
        df["return"] = df["CLOSE"].pct_change()
        neg_return = np.where(df["return"]>0,0,df["return"])
        neg_vol = pd.Series(neg_return[neg_return!=0]).std() * np.sqrt(252)
        return (self.CAGR(df) - rf)/neg_vol