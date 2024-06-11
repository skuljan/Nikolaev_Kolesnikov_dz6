# -*- coding: utf-8 -*-
# дз2 номер 6. Выполнили: Николаев Ян, Колесников Илья

import zipfile
import os
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from isoweek import Week
import matplotlib.pyplot as plt
import logging
import torch.multiprocessing as mp

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data():
    # Распаковка архива
    with zipfile.ZipFile('rossmann.zip', 'r') as zip_ref:
        zip_ref.extractall('')
    logging.info('Данные распакованы.')

    # Имена файлов
    table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']
    tables = [pd.read_csv(fname + '.csv', low_memory=False) for fname in table_names]
    train, store, store_states, state_names, googletrend, weather, test = tables
    logging.info('Таблицы загружены.')

    # Объединение таблиц
    def join_df(left, right, left_on, right_on=None):
        if right_on is None: right_on = left_on
        return left.merge(right, how='left', left_on=left_on, right_on=right_on, suffixes=("", "_y"))

    weather = join_df(weather, state_names, "file", "StateName")
    googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]
    googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]
    googletrend.loc[googletrend.State == 'NI', "State"] = 'HB,NI'

    # Преобразование даты
    def add_datepart(df):
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Week"] = df["Date"].dt.isocalendar().week
        df["Day"] = df["Date"].dt.day

    add_datepart(weather)
    add_datepart(googletrend)
    add_datepart(train)
    add_datepart(test)
    logging.info('Дата преобразована.')

    # Объединение данных
    trend_de = googletrend[googletrend.file == 'Rossmann_DE']
    store = join_df(store, store_states, "Store")
    joined = join_df(train, store, "Store")
    joined = join_df(joined, googletrend, ["State", "Year", "Week"])
    joined = joined.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
    joined = join_df(joined, weather, ['State', 'Date'])
    logging.info('Данные объединены.')

    # Обработка данных
    joined.CompetitionOpenSinceYear = joined.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)
    joined.CompetitionOpenSinceMonth = joined.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)
    joined.Promo2SinceYear = joined.Promo2SinceYear.fillna(1900).astype(np.int32)
    joined.Promo2SinceWeek = joined.Promo2SinceWeek.fillna(1).astype(np.int32)

    joined["CompetitionOpenSince"] = pd.to_datetime(joined.apply(lambda x: datetime.datetime(
        x.CompetitionOpenSinceYear, x.CompetitionOpenSinceMonth, 15), axis=1))
    joined["CompetitionDaysOpen"] = joined.Date.subtract(joined["CompetitionOpenSince"]).dt.days
    joined["CompetitionDaysOpen"] = joined["CompetitionDaysOpen"].apply(lambda x: x if x >= 0 else 0)
    joined["CompetitionMonthsOpen"] = joined["CompetitionDaysOpen"] // 30
    joined["CompetitionMonthsOpen"] = joined["CompetitionMonthsOpen"].apply(lambda x: x <= 24 and x or 24)

    joined["Promo2Since"] = pd.to_datetime(
        joined.apply(lambda x: Week(x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1))
    joined["Promo2Days"] = joined.Date.subtract(joined["Promo2Since"]).dt.days
    joined["Promo2Days"] = joined["Promo2Days"].apply(lambda x: x if x >= 0 else 0)
    joined["Promo2Weeks"] = joined["Promo2Days"] // 7
    joined["Promo2Weeks"] = joined["Promo2Weeks"].apply(lambda x: x <= 25 and x or 25)
    logging.info('Промо и конкуренция обработаны.')

    # Кодирование категориальных переменных
    cat_var_dict = {
        'Store': 50, 'DayOfWeek': 6, 'Year': 2, 'Month': 6, 'Day': 10, 'StateHoliday': 3,
        'StoreType': 2, 'Assortment': 3, 'PromoInterval': 3, 'State': 6,
        'Week': 2, 'CompetitionOpenSinceYear': 4, 'Promo2SinceYear': 4
    }
    cat_vars = list(cat_var_dict.keys())
    contin_vars = [
        'CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
        'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h',
        'CloudCover', 'trend', 'trend_DE', 'CompetitionMonthsOpen', 'Promo2Weeks', 'Promo', 'SchoolHoliday'
    ]

    for v in contin_vars: joined[v] = joined[v].fillna(0)
    for v in cat_vars: joined[v] = joined[v].fillna("").astype(str)

    # Преобразование категориальных переменных в числовые
    label_encoders = {col: LabelEncoder().fit(joined[col]) for col in cat_vars}
    for col, le in label_encoders.items():
        joined[col] = le.transform(joined[col])

    # Нормализация непрерывных переменных
    scalers = {col: StandardScaler().fit(joined[[col]]) for col in contin_vars}
    for col, scaler in scalers.items():
        joined[col] = scaler.transform(joined[[col]])

    logging.info('Категориальные и непрерывные переменные обработаны.')

    # Удаление всех случаев с нулевыми продажами
    joined = joined[joined.Sales != 0]

    # Разделение данных на тренировочные и валидационные
    train_ratio = 0.9
    train_size = int(len(joined) * train_ratio)

    joined_train = joined.iloc[:train_size]
    joined_valid = joined.iloc[train_size:]

    X_train_cat = joined_train[cat_vars].values
    X_valid_cat = joined_valid[cat_vars].values

    X_train_cont = joined_train[contin_vars].values
    X_valid_cont = joined_valid[contin_vars].values

    y_train = np.log(joined_train.Sales.values)
    y_valid = np.log(joined_valid.Sales.values)

    logging.info('Данные разделены на тренировочные и валидационные.')

    return (X_train_cat, X_train_cont, y_train), (X_valid_cat, X_valid_cont, y_valid), label_encoders, contin_vars, cat_vars


# Подготовка данных
class RossmannDataset(Dataset):
    def __init__(self, cat_data, cont_data, targets=None):
        self.cat_data = torch.tensor(cat_data, dtype=torch.long)
        self.cont_data = torch.tensor(cont_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.targets is not None:
            return (self.cat_data[idx], self.cont_data[idx]), self.targets[idx]
        else:
            return self.cat_data[idx], self.cont_data[idx]


# Определение модели
class RossmannModel(nn.Module):
    def __init__(self, emb_szs, n_cont):
        super(RossmannModel, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(0.02)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        self.layers = nn.Sequential(
            nn.Linear(sum([nf for ni, nf in emb_szs]) + n_cont, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout(0.2),
            nn.Linear(500, 1)
        )

    def forward(self, x_cat, x_cont):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x


def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            cat_data, cont_data = inputs
            optimizer.zero_grad()
            outputs = model(cat_data, cont_data)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * cat_data.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                cat_data, cont_data = inputs
                outputs = model(cat_data, cont_data)
                loss = criterion(outputs.squeeze(), targets)
                valid_loss += loss.item() * cat_data.size(0)

        epoch_valid_loss = valid_loss / len(valid_loader.dataset)

        logging.info(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}')


def main():
    # Подготовка данных
    (X_train_cat, X_train_cont, y_train), (X_valid_cat, X_valid_cont, y_valid), label_encoders, contin_vars, cat_vars = prepare_data()

    # Подготовка датасетов и загрузчиков данных
    train_dataset = RossmannDataset(X_train_cat, X_train_cont, y_train)
    valid_dataset = RossmannDataset(X_valid_cat, X_valid_cont, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4)
    logging.info('Датасеты и загрузчики данных подготовлены.')

    # Подготовка модели
    emb_szs = [(len(label_encoders[col].classes_), min(50, (len(label_encoders[col].classes_) + 1) // 2)) for col in cat_vars]
    n_cont = len(contin_vars)

    model = RossmannModel(emb_szs, n_cont)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    logging.info('Начало тренировки модели.')

    # Обучение модели
    train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=5)
    logging.info('Тренировка модели завершена.')

    # Оценка модели
    def mape(preds, y_true):
        return np.mean(np.abs((y_true - preds) / y_true)) * 100

    model.eval()
    with torch.no_grad():
        valid_preds = []
        for inputs, _ in valid_loader:
            cat_data, cont_data = inputs
            outputs = model(cat_data, cont_data)
            valid_preds.append(outputs.squeeze().numpy())

        valid_preds = np.concatenate(valid_preds)
        valid_preds = np.exp(valid_preds)
        metric = mape(valid_preds, np.exp(y_valid))
        logging.info(f'MAPE: {metric:.4f}%')

    # Визуализация результатов
    n0 = 100
    n1 = 200

    x = list(range(n1 - n0))
    plt.figure(figsize=(12, 4))
    plt.plot(x, np.exp(y_valid[n0:n1]), label='Actual')
    plt.plot(x, valid_preds[n0:n1], label='Predicted')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
