# Гиперпараметры для самописного трансформера

seed = 123
epochs = 5
max_encoder_length = 28 * 7
min_encoder_length = max_encoder_length // 2
prediction_length = 28
batch_size = 256
learning_rate = 0.001
transformer_size = 128
transformer_hidden_dim = 128
num_encoder_layers = 3
num_decoder_layers = 3
attention_heads = 8

categorical_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
real_columns = ['day', 'time_of_year', 'day_of_week']

SAVE_PATH = './models/CustomModel_'
