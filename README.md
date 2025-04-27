# 🚀 Predictive Modeling for Stock Price Prediction using Time Series & Causal Analysis

## 📌 Table of Contents

- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Model Details](...)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction

A brief overview of your project and its purpose. Mention which problem statement are your attempting to solve. Keep it
concise and engaging.

## 🎥 Demo

🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## 💡 Inspiration

The stock market is volatile with factors that can affect its performance, making it difficult to predict. Whereas, as
institutions employ sophisticated analytics, retail investors work with little raw data. Most existing models either
capture trends over time or the causal influences without consistent profitability.

This project integrates time-series analysis with causal inference in order to improve prediction accuracy by modeling
cause-and-effect relationships in stock price movements. The machine learning approach provides a new avenue to address
challenges such as noise and non-stationarity given the advancement of machine learning and increasing access to vast
amount of financial data, eventually making it a highly relevant and relatable problem for the team.

## ⚙️ What It Does

Objective of this project is to build a predictive model for stock prices by combining time series forecasting and
causal analysis to understand key factors influencing stock market trends. The expected outcomes of this project:

- A predictive model for stock price forecasting using time series.
- A graph-based model incorporating cross-sectoral dependencies for enhanced prediction.
- A comprehensive report detailing model performance and key causal factors identified.

### Use cases:

- Cross-sectoral analysis of stock dependencies for portfolio optimization.
- Enhanced prediction of price movements with causal relationships.
- Advanced risk assessment for institutional and retail investors.

### HLD & Workflow

![image](https://github.com/user-attachments/assets/7f632bf8-044f-43f9-9ad7-67471465b2f8)

## 🛠️ How We Built It

Briefly outline the technologies, frameworks, and tools used in development.

## Model Details

## Background

Traditional models like LSTMs often fail in stock market prediction due to their
inability to capture long-term dependencies, noisy data, and the non-stationary
nature of financial time series. This emphasizes the need for advanced architectures like GNNs and
Transformers that can better handle complex dependencies and causal relationships.

## Model Details

### LSTM - Long Short Term Memory

<details>
   <summary>🔍 Click to see details.</summary>

LSTMs, are a specialized type of RNN architecture designed to tackle a specific challenge—remembering information over
extended periods. These models that enhance the memory capabilities of recurrent neural networks. These networks
typically hold short-term memory, utilizing earlier information for immediate tasks within the current neural network.

LSTMs can analyze historical price data and past events to potentially predict future trends, considering long-term
factors that might influence the price.
Even though LSTMs offer advantages for predicting stock market prices, there are still challenges to consider:

**Data Quality and Noise**: A multitude of factors influences stock prices, many of which remain unpredictable, such as
news events and social media sentiment. LSTMs might struggle to differentiate between relevant patterns and random noise
in
the data, potentially leading to inaccurate predictions.

**Limited Historical Data**: The effectiveness of LSTMs depends on the quality and quantity of historical data
available.
For newer companies or less liquid stocks, there might not be enough data to train the model effectively, limiting its
ability to capture long-term trends.

**Non-Linear Relationships**: While LSTMs can handle complex relationships, the stock market can exhibit sudden shifts
and
non-linear behavior due to unforeseen events. The model might not be able to perfectly capture these unpredictable
fluctuations.

**Overfitting and Generalizability**: There’s a risk of the model overfitting the training data, performing well on
historical data but failing to generalize to unseen future patterns. Careful hyperparameter tuning and validation
techniques are crucial to ensure the model can learn generalizable insights.

**Self-Fulfilling Prophecies**: If a large number of investors rely on LSTM predictions, their collective actions could
influence the market in a way that aligns with the prediction, creating a self-fulfilling prophecy. This highlights the
importance of using these predictions as a potential guide, not a guaranteed outcome.

**Causal Factors**: LSTM cannot take advantage of causal analysis in predicting a stock price. Macro factors such as
USD-INR, RepoRate etc play significant role in causing stock price movements

#### LSTM Results

When trained with Indian stock price details along with its macro-factors in place with the data, the
LSTM struggled to incorporate the causal dependencies and non-linear dependencies.

**Model Details**

```plain
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                     │ (None, 64)             │        18,688 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 1)              │            65 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 56,261 (219.77 KB)
 Trainable params: 18,753 (73.25 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 37,508 (146.52 KB)
```

**Model Metric**

```plain
R-squared (R2): 0.8507604692671287
```

**Conclusion**
Since R2 is not very great, also, the model loss during training is found to be too high.
Hence LSTM is not a good fit for stock target prediction.

</details>

## 🚧 Challenges We Faced

Describe the major technical or non-technical challenges your team encountered.

## 🏃 How to Run

1. Clone the repository
   ```sh
   https://github.com/vchandraiitk/iisc-capstone.git
   ```
2. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```
3. Run the project
   ```sh
   python app.py
   ```

## 🏗️ Tech Stack

- 🔹 Frontend: Streamlit/Gradio
- 🔹 Backend: Python
- 🔹 Database: csv files
- 🔹 Other: Fill

## 👥 Team

- **Madhava Keshavamurthy
  ** - [GitHub](https://github.com/madhavamk) | [LinkedIn](https://www.linkedin.com/in/madhavamk/)
- **Vikas Chandra
  ** - [GitHub](https://github.com/vchandraiitk) | [LinkedIn](https://www.linkedin.com/in/vikas-chandra-3112496/)
- **Richa Parikh
  ** - [GitHub](https://github.com/richa1543) | [LinkedIn](https://www.linkedin.com/in/richa-parikh-824a1b124/)

<img width="697" alt="image" src="https://github.com/user-attachments/assets/09307f59-6cb1-46ae-b237-e5e3c87fa078" />

