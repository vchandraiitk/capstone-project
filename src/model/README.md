# For GNN-GAT with corss sector dependency 
## 🔷 1. Model Hyperparameters (GATConv + Linear)
### 🔸 in_channels
What: Number of input features per node in the graph.

Set as: graph_data.num_node_features

Effect: Determines the dimensionality of the input layer of the GAT.

Tuning: Controlled indirectly via feature engineering — more features = higher input size.

### 🔸 out_channels = 16 (in GATConv)
What: Output size per attention head.

Effect: Controls how richly each node is represented internally.

Tuning Tip: Try 8, 32, 64 to adjust model capacity.

### 🔸 heads = 4
What: Number of attention heads in GAT layer.

Effect: Multiple heads allow the model to learn from different attention perspectives.

Tuning Tip: 2–8 are common values. More heads → richer aggregation but higher memory usage.

### 🔸 dropout = 0.3
What: Dropout probability during attention layer computation.

Effect: Reduces overfitting. Applied to attention coefficients.

Tuning Tip: Try 0.1–0.5. Higher dropout helps regularize with small datasets.

### 🔸 Linear(16 * 4, 1)
What: Final layer mapping from GAT output to a single regression output.

Input size: out_channels * heads = 16 * 4 = 64

Output: Scalar prediction (Close_scaled)

## 🔷 2. Training Hyperparameters
### 🔸 epochs = 100
What: Number of complete training loops over the dataset.

Effect: More epochs can lead to better fit — but risk overfitting.

Tuning Tip: Use validation loss and early stopping for smarter convergence.

### 🔸 lr = 0.01 (learning rate in Adam)
What: Controls the step size in weight updates.

Effect: Too high → unstable training; too low → slow convergence.

Tuning Tip: Try 0.001, 0.005, 0.02 — or schedule/decay learning rate.

### 🔸 optimizer = Adam(model.parameters(), lr=0.01)
What: Optimization algorithm.

Effect: Adam adapts learning rate per parameter; handles sparse gradients well.

Alternatives: Try AdamW or RMSProp.

### 🔸 loss_fn = MSELoss()
What: Mean Squared Error between predicted and actual scaled price.

Effect: Penalizes large errors more heavily.

Alternatives: SmoothL1Loss is robust to outliers.

## 🔷 3. Forecasting / Uncertainty Estimation
### 🔸 n_samples = 50 (in forecast_future_prices)
What: Number of stochastic forward passes for MC Dropout.

Effect: More samples = better confidence estimates.

Tuning Tip: 100 or 200 for more stable variance estimates (at compute cost).

### 🔸 horizons = [30, 180, 365]
What: Future time points (in days) to forecast for each stock.

Effect: Affects how far into the future you're predicting.

Tuning Tip: Add [60, 90] for mid-range insights.

## 🔷 4. Graph Building Parameters
### 🔸 Granger overlap logic
What: Uses shared causal factors to form edges between tickers.

Effect: Influences graph connectivity → affects message passing.

Tuning Tip: You can introduce a threshold for overlap strength or edge weight.

### 🔸 cross_sector_edges
What: Hardcoded mapping of related industries.

Effect: Adds inter-industry edges for knowledge flow.

Tuning Tip: Expand with real-world sector interactions or use cosine similarity.

## 🔷 5. Feature Engineering (affects model indirectly)
Features used:
Close_scaled

Close_scaled_diff

time_index_scaled

Granger-caused factors (scaled)

Industry (one-hot)

All features influence in_channels, and their selection plays a major role in model performance.

## ✅ Summary Table
Hyperparameter	Category	Description	Tuning Tip
out_channels=16	GAT	Embedding size per head	Try 8, 32
heads=4	GAT	Number of attention heads	Try 2–8
dropout=0.3	GAT	Prevents overfitting	Try 0.1–0.5
lr=0.01	Optimizer	Learning rate	Try 0.001–0.02
epochs=100	Training	Epochs for training	Use early stopping
loss_fn=MSE	Loss	Penalizes squared errors	Try SmoothL1Loss
n_samples=50	Forecasting	MC Dropout forward passes	Try 100 for stability
horizons=[30,180,365]	Forecasting	Forecast days ahead	Add 60, 90 for mid-term
