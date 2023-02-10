# Файл с обучением самописного трансформера
# Трансформер обучается очень медленно (~14 часов на одну эпоху на GTX 3060) используя 1/20 часть датасета М5
# пока руки не дошли до ускорения его обучения

import torch
import numpy as np
import pandas as pd
import tqdm
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
import tensorflow as tf
import tensorboard as tb
from helper_functions import generate_square_subsequent_mask

from custom_transformer_hyperparameters import *
from models import Transformer

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
torch.set_float32_matmul_precision('medium')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Функция для оценки работы модели
# Отмечается, что MAE ошибки трансформера и наивной модели примерно одинаковые (~1.2), однако медианная ошибка в
# модели трансформера практический нулевая, тем временем как она составляет ~0.53 в наивном алгоритме
def test_model(model, data_series):
    dataloader = data_series.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    all_errors = []
    all_targets = []
    for input_data, (targets, weights) in iter(dataloader):
        model_predictions = model(input_data)
        model_errors = (targets - model_predictions).abs()
        all_targets.append(targets.numpy())
        all_errors.append(model_errors.numpy())

    all_errors = np.concatenate(all_errors)
    all_targets = np.concatenate(targets)
    # Средние и медианы ошибок по разным перцинтилям
    print(f"Total MAE loss:", all_errors.mean())
    print(f"Total MAE loss median:", np.median(all_errors))
    print(pd.DataFrame(all_errors).describe().mean(axis=1))

    naive_predictions = np.roll(all_targets, 1, dims=-1)
    naive_errors = (all_targets - naive_predictions).abs()
    print("Naive loss:", naive_errors.mean())
    print("Naive loss median:", naive_errors.mean())
    print(pd.DataFrame(naive_errors).describe().mean(axis=1))
    return all_errors.mean(), all_errors.median()


if __name__ == "__main__":
    data = pd.read_csv('./datasets/train_val_part.csv', index_col=0)
    data['sales'] = data['sales'].astype(np.float32)
    results = {}

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_val_split_point = data['day'].max() - (prediction_length * 10)

    # Создаем дата-генератор используя готовые варианты
    train_subset = TimeSeriesDataSet(data=data[lambda x: x.day <= train_val_split_point],
                                     time_idx='day',
                                     target='sales',
                                     group_ids=categorical_columns,
                                     min_encoder_length=min_encoder_length,
                                     max_encoder_length=max_encoder_length,
                                     min_prediction_length=1,
                                     max_prediction_length=prediction_length,
                                     static_categoricals=categorical_columns,
                                     time_varying_known_reals=real_columns,
                                     time_varying_unknown_reals=['sales'],
                                     target_normalizer=GroupNormalizer(
                                         groups=categorical_columns,
                                         transformation="softplus"),
                                     add_relative_time_idx=True,
                                     add_target_scales=True,
                                     add_encoder_length=True
                                     )

    # Создаем дата-генератор для валидации
    val_subset = TimeSeriesDataSet(data=data[lambda x: x.day > (train_val_split_point - min_encoder_length)],
                                   time_idx='day',
                                   target='sales',
                                   group_ids=categorical_columns,
                                   min_encoder_length=max_encoder_length,
                                   max_encoder_length=max_encoder_length,
                                   min_prediction_length=prediction_length,
                                   max_prediction_length=prediction_length,
                                   static_categoricals=categorical_columns,
                                   time_varying_known_reals=real_columns,
                                   time_varying_unknown_reals=['sales'],
                                   target_normalizer=GroupNormalizer(
                                     groups=categorical_columns,
                                     transformation="softplus"),
                                   add_relative_time_idx=True,
                                   add_target_scales=True,
                                   add_encoder_length=True
                                   )

    train_dataloader = train_subset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = val_subset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # Создаем модель, оптимизатор, и задаем функцию ошибки
    model = Transformer(data, categorical_columns, real_columns, target_size=prediction_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    # Создаем маски для декодера
    src_mask = generate_square_subsequent_mask(prediction_length, prediction_length)
    target_mask = generate_square_subsequent_mask(prediction_length, max_encoder_length)

    epoch_train_losses = []  # Список в котором храним ошибки на обучающей выборке
    epoch_val_losses = []    # Список в котором храним ошибки на валидационной выборке

    for epoch in range(epochs):
        model.train()
        batch_train_losses = []

        # Обучающая эпоха
        for batch in tqdm.tqdm(iter(train_dataloader)):
            optimizer.zero_grad()
            input_data = batch[0]
            targets = batch[0]['decoder_target'].to(device)
            outputs = model(input_data, src_mask, target_mask).to(device)
            train_loss = loss_function(outputs, targets)
            batch_train_losses.append(float(train_loss))

            train_loss.backward()
            optimizer.step()

        epoch_train_loss = np.mean(batch_train_losses)
        print(f"Average loss for train epoch {epoch+1}: {epoch_train_loss}")

        model.eval()
        batch_val_losses = []

        # Валидационная эпоха
        for batch in tqdm.tqdm(iter(val_dataloader)):
            input_data = batch[0]
            targets = batch[0]['decoder_target'].to(device)
            outputs = model(input_data).to(device)
            val_loss = loss_function(outputs, targets)
            batch_val_losses.append(float(val_loss))

        epoch_val_loss = np.mean(batch_val_losses)
        print(f"Average loss for validation epoch {epoch+1}: {epoch_val_loss}")

        epoch_train_losses.append(epoch_train_loss)
        epoch_val_losses.append(epoch_val_loss)

        # Сохраняем обучение после каждой эпохи
        SAVE_STRING = SAVE_PATH+f'CUSTOM_TRANSFORMER_SMALL' \
                                f'_Epoch={epoch}_TrainLoss={round(float(epoch_train_loss), 4)}' \
                                f'_ValLoss={round(float(epoch_train_loss), 4)}'

        model.save(model.state_dict(), SAVE_STRING)
        mean, median = test_model(model, data_series=val_subset)
        results[f'Epoch_{epoch}'] = f"Mean loss: {mean}, Median loss: {median}"
    print(results)
