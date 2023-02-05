# Главный файл проекта, в котором реализован алгоритм Temporal Fusion Transformer (В основном использует готовые модули)
# Обучение алгоритма занимает ~4 часа на GTX 3060 используя лишь 1/20 часть датасета М5.
# Работа обученного трансформера сравнивается с "наивным" алгоритмом, который лишь выдает предыдущее значение в качестве предсказания.
# Ошибка MAE обученной TFT модели сопоставимо с наивным алгоритмом, однако более глубокий анализ показывает,
# что TFT-модель ошибается в основном на выбросах (см. функция test_model).
# Предполагается, что если добавить больше полезных фичей в датасет (например, наличие праздников), TFT-модель будет работать значительно лучше.

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import tensorflow as tf
import tensorboard as tb

from TFT_hyperparameters import *
from helper_functions import visualize_function, transform_values

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
torch.set_float32_matmul_precision('medium')

# Колбэки
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=1, verbose=True, mode="min")
lr_logger = LearningRateMonitor()
checkpoint_callback = ModelCheckpoint(dirpath='./models', filename='{epoch}-{val_loss:.2f}-{train_loss_epoch:.2f}')
logger = TensorBoardLogger("lightning_logs")


# Функция, которая создает модель трансформера
def get_transformer(train_series):
    model = TemporalFusionTransformer.from_dataset(
        train_series,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=len(quantiles_list),
        loss=QuantileLoss(quantiles=quantiles_list),
        log_interval=10,
        reduce_on_plateau_patience=4)

    return model


# Функция для обучения модели
def train_model(train_series, val_series):
    train_dataloader = train_series.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = val_series.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu',
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[checkpoint_callback, lr_logger, early_stop_callback],
        logger=logger,
        default_root_dir="./models/")

    model = get_transformer(train_series)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader)

    return model


# Функция для оценки работы модели
# Отмечается, что MAE ошибки трансформера и наивной модели примерно одинаковые (~1.2), однако медианная ошибка в
# модели трансформера практический нулевая, тем временем как она составляет ~0.53 в наивном алгоритме
def test_model(model, data_series):
    dataloader = data_series.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    targets = torch.cat([y for x, (y, weight) in iter(dataloader)])
    predictions = model.predict(dataloader, mode='raw')['prediction']
    for i in range(len(quantiles_list)):
        model_predictions = predictions[..., i]
        model_errors = (targets - model_predictions).abs()
        # Средние и медианы ошибок по разным перцинтилям
        print(f"Model P{quantiles_list[i]} MAE loss:", model_errors.mean())
        print(f"Model P{quantiles_list[i]} MAE loss median:", np.median(model_errors))
        print(pd.DataFrame(model_errors).describe().mean(axis=1))

    naive_predictions = torch.roll(targets, 1, dims=-1)
    naive_errors = (targets - naive_predictions).abs()
    print("Naive loss:", naive_errors.mean())
    print("Naive loss median:", naive_errors.mean())
    print(pd.DataFrame(naive_errors).describe().mean(axis=1))


# Функция для создания и загрузки весов модели с сохраненного чек пойнта
def load_model(checkpoint, train_series):
    model = get_transformer(train_series)

    model.load_from_checkpoint(checkpoint)
    return model


# Основной код - обучение и тестирование модели
if __name__ == "__main__":
    data = pd.read_csv('./datasets/train_val_part.csv', index_col=0)
    data['sales'] = data['sales'].astype(np.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_val_split_point = data['day'].max() - max_prediction_length

    train_subset = TimeSeriesDataSet(data=data[lambda x: x.day <= train_val_split_point],
                                     time_idx='day',
                                     target='sales',
                                     group_ids=categorical_columns,
                                     min_encoder_length=max_encoder_length//2,
                                     max_encoder_length=max_encoder_length,
                                     min_prediction_length=1,
                                     max_prediction_length=max_prediction_length,
                                     static_categoricals=categorical_columns,
                                     time_varying_known_reals=real_columns,
                                     time_varying_unknown_reals=['sales'],
                                     target_normalizer=GroupNormalizer(groups=categorical_columns, transformation="softplus"),
                                     add_relative_time_idx=True,
                                     add_target_scales=True,
                                     add_encoder_length=True
                                     )

    val_subset = TimeSeriesDataSet.from_dataset(train_subset, data, predict=True, stop_randomization=True)

    transformer = train_model(train_subset, val_subset)

    # transformer = load_model('./models/epoch=2-val_loss=0.47-train_loss_epoch=0.38.ckpt', train_series=train_subset)
    test_model(transformer, data_series=val_subset)

    # input_data = transform_values(data.iloc[0:num_days, :], val_subset)
    # visualize_function(0, data, subset=val_subset, model=transformer)

