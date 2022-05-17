# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:44:50 2022

@author: HSelato
"""

import numpy as np
import pandas as pd

################################Backtesting####################################

class Backtesting:
#rebalancing of portofolio monthly
# function to calculate portfolio return iteratively for long position as exercise do the short part
    def pflio(self, DF,m,x,period):
        """Returns cumulative portfolio return
        DF = dataframe with monthly return info for all stocks
        m = number of stock in the portfolio
        x = number of underperforming stocks to be removed from portfolio monthly"""
        df = DF.copy()
        portfolio = []
        monthly_ret = [0]
        for i in range(len(df)):
            if len(portfolio) > 0:
                monthly_ret.append(df[portfolio].iloc[i,:].mean())
                bad_stocks = df[portfolio].iloc[i,:].sort_values(ascending=True)[:x].index.values.tolist()
                portfolio = [t for t in portfolio if t not in bad_stocks]
            fill = m - len(portfolio)
            new_picks = df.iloc[i,:].sort_values(ascending=False)[:fill].index.values.tolist()
            portfolio = portfolio + new_picks
            print(portfolio)
        monthly_ret_df = pd.DataFrame(np.array(monthly_ret),columns=[period+"_ret"])
        return monthly_ret_df