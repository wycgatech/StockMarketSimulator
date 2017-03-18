## README

This project implement a stock market simulator. The ultimate goal is to let the simulator generate the best trading strategy automatically and tell me which stock I should buy or sell to make the most benefit. 

The project contains three different major components:  

- A Market Simulator: Given a portfolio (which is like a person account that has information about what stocks are you currently holding) and an order book (a series of BUY/SELL orders), return the estimated result.  

- Machine Learning Models: A stock price can be predicted in two ways, we can learn patterns from historical price data, which is a pure quantitative way to train model, and we can learn from up to date news to see if the company is doing well or not and how their stock price is gonna change. 

  For the first part I applied a few machine learning models such as linear regression, random forest and Q learning to train models based on historical data. The second part is natural language processing, which haven't not been implemented yet. 

- Trading Strategy Generator: Apply the trained learning models and covert it to a classification learner using different technical indicators (e.g. Exponential Moving Average, Bollinger Band, Relative Strength Index and so on). The classification learners is then used to determine a series of BUY/SELL orders to form the best trading strategy. 