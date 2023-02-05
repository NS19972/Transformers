import tqdm
import numpy as np
import pandas as pd
import torch

num_days = 1913 # Количество дней в датасете

# Функция, которая читает файл sales_train_validation и обрабатывает датасет в нужный формат.
# В результате создается .csv файл весом 3.1 гигабайт, который затем читается в main.py
# Содержимые файла используются в генераторе.
def convert_dataset():
    data = pd.read_csv('./datasets/sales_train_validation.csv')
    with tqdm.tqdm(total=data.shape[0]) as progress_bar:
        row_index = 0
        data_list = []
        while row_index < data.shape[0]:
            if not row_index % 20 == 0:  # Берём только каждый двадцатый отчет (чтобы датасет не был слишком большим)
                row_index += 1
                progress_bar.update(1)
                continue
            temp = pd.DataFrame({'day': np.arange(num_days),
                                 'day_of_week': np.arange(num_days) % 7,
                                 'time_of_year': np.arange(num_days) / 365 - np.arange(num_days) // 365,
                                 'sales': data.iloc[row_index, data.shape[1]-num_days:].values,
                                 'item_id': np.repeat(data.iloc[row_index]['item_id'], num_days),
                                 'dept_id': np.repeat(data.iloc[row_index]['dept_id'], num_days),
                                 'cat_id': np.repeat(data.iloc[row_index]['cat_id'], num_days),
                                 'store_id': np.repeat(data.iloc[row_index]['store_id'], num_days),
                                 'state_id': np.repeat(data.iloc[row_index]['state_id'], num_days)},
                                index=np.arange(num_days))

            data_list.append(temp)
            row_index += 1
            progress_bar.update(1)

    dataframe = pd.concat(data_list, axis=0).reset_index(drop=True)
    dataframe.to_csv('./datasets/train_val_part.csv')


# Функция для нормирования всех данных для нейросети (по тем же правилам, что делает объект TimeSeriesDataset)
def transform_values(data, data_series):
    transformed_data = []
    for column in data.columns:
        transformed_data.append(data_series.transform_values(name=column, values=data[column].values))
    transformed_data = torch.cat(transformed_data, dim=-1)
    return transformed_data


# Функция для визуализации предсказания (не дописанная)
def visualize_function(item_index, data, subset, model):
    input_data = transform_values(data[item_index*num_days: (item_index+1):num_days], subset)
    result = model.predict(input_data)
    return result


# Функция для генерирования масок для самописного трансформера (взята с интернета)
def generate_square_subsequent_mask(dim1: int, dim2: int) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Source:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e.
              the length of the input sequence to the model),
              and for tgt masking, this must be target sequence length
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


if __name__ == '__main__':
    convert_dataset()
