# Гиперпараметры для модели Temporal Fusion Transform
seed = 123
max_encoder_length = 28 * 7
max_prediction_length = 28
batch_size = 128
quantiles_list = [0.05, 0.1, 0.5, 0.9, 0.95]
learning_rate = 0.001
hidden_size = 256
attention_head_size = 8
dropout = 0.05
hidden_continuous_size = 128
reduce_on_plateau_patience = 4

categorical_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
real_columns = ['day', 'day_of_week']
