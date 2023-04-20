import pickle
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import streamlit as st


def main():
    model = load_model("model/logist_regress(2).pkl")
    test_data_X, test_data_y, dt = load_test_data("data/weather_AUS_preprocessed.csv")
    
    page = st.sidebar.selectbox(
        "Выберите страницу",
        ["Описание задачи и данных", "Запрос к модели"]
    )

    if page == "Описание задачи и данных":
        st.title("Описание задачи и данных")
        st.write("Выберите страницу слева")

        st.header("Описание задачи")
        st.markdown("""Задача данной модели машинного обучения – предсказание дождей в Австралии""")

        st.header("Описание данных")
        st.markdown("""Этот набор данных содержит около 10 лет ежедневных наблюдений за погодой из многих мест по всей Австралии.
* Date – дата наблюдения,
* Location – общее название места расположения метеостанции,
* MinTemp – минимальная температура в градусах Цельсия,
* MaxTemp – максимальная температура в градусах Цельсия,
* Rainfall – количество осадков, выпавших за сутки в мм,
* Evaporation – так называемое испарение класса А (мм) за 24 часа до 9 утра,
* Sunshine – количество часов яркого солнечного света в сутки,
* WindGustDir – направление самого сильного порыва ветра за 24 часа до полуночи,
* WindGustSpeed – скорость (км/ч) самого сильного порыва ветра за 24 часа до полуночи,
* WindDir9am – направление ветра в 9 утра,
* WindDir3pm – направление ветра в 3 часа дня,
* WindSpeed9am – средняя скорость ветра (км/ч) за 10 минут до 9 утра,
* WindSpeed3pm – средняя скорость ветра (км/ч) за 10 минут до 3 часов дня,
* Humidity9am – влажность (в процентах) в 3 часа дня,
* Humidity3pm – влажность (в процентах) в 9 утра,
* Pressure9am – атмосферное давление (гПа) сниженное до среднего уровня моря в 9 утра,
* Pressure3pm – атмосферное давление (гПа) снизилось до среднего уровня моря в 3 часа дня,
* Cloud9am – облачность в 9 утра в "октах", изменяется от 1 до 8,
* Cloud3pm – облачность в 3 часа дня в "октах", изменяется от 1 до 8,
* Temp9am – температура в градусах Цельсия в 9 утра,
* Temp3pm – температура в градусах Цельсия в 3 часа дня,
* RainToday – булево значение: 1, если осадки (мм) за 24 часа до 9 утра превышают 1 мм, в противном случае 0,
* RainTomorrow – булево значение: 1, если осадки (мм) за 24 часа до 9 утра следующего дня превышают 1 мм, в противном случае 0.
""")

    elif page == "Запрос к модели":
        st.title("Запрос к модели")
        st.write("Выберите страницу слева")
        request = st.selectbox(
            "Выберите запрос",
            ["Accuracy", "Первые 5 предсказанных значений", "Пользовательский пример"]
        )

        if request == "Accuracy":
            st.header("Точность модели")
            accuracy = model.score(test_data_X, test_data_y)
            st.write(f"{accuracy}")
        elif request == "Первые 5 предсказанных значений":
            st.header("Первые 5 предсказанных значений")
            first_5_test = test_data_X[:5, :]
            first_5_pred = model.predict(first_5_test)
            for item in first_5_pred:
                st.write(f"{item:.2f}")
        elif request == "Пользовательский пример":
            st.header("Пользовательский пример")

            location = st.selectbox("Местоположение", ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
       'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
       'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
       'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
       'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
       'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
       'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
       'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
       'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
       'AliceSprings', 'Darwin', 'Katherine', 'Uluru'])
            str_loc = "Location_"+location
            dt[str_loc] = 1

            dt["MaxTemp"] = st.number_input("Максимальная температура", -10, 60)

            dt["MinTemp"] = st.number_input("Минимальная температура", -10, 60)

            dt["Rainfall"] = st.number_input("Количество осадков", 0, 300)

            dt["Evaporation"] = st.number_input("Испарение", 0, 1000)

            dt["Sunshine"] = st.number_input("Количество часов солнечного света", 0, 124)

            wind_gust_dir = st.selectbox("Направление самого сильного порыва ветра", ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW','ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'])
            str_loc = "WindGustDir_"+wind_gust_dir
            dt[str_loc] = 1

            dt["WindGustSpeed"] = st.number_input("Скорость самого сильного порыва ветра", 0, 100)

            wind_dir9 = st.selectbox("Направление ветра в 9 утра", ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW' 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'])
            str_loc = "WindDir9am_"+wind_dir9
            dt[str_loc] = 1

            dt["WindSpeed9am"] = st.number_input("Скорость ветра в 9 утра", 0, 100)

            wind_dir3 = st.selectbox("Направление ветра в 3 часа дня", ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW','ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'])
            str_loc = "WindDir3pm_"+wind_dir3
            dt[str_loc] = 1

            dt["WindSpeed3pm"] = st.number_input("Скорость ветра в 3 часа дня", 0, 100)

            dt["Humidity9am"] = st.number_input("Влажность в 9 утра", 0, 100)

            dt["Humidity3pm"] = st.number_input("Влажность в 3 часа дня", 0, 100)

            dt["Pressure9am"] = st.number_input("Атмосферное давление в 9 утра", 80, 120)

            dt["Cloud9am"] = st.number_input("Облачность в 9 утра", 1, 8)

            dt["Cloud3pm"] = st.number_input("Облачность в 3 часа дня", 1, 8)

            dt["RainToday"] = st.selectbox("Идёт ли сегодня дождь", ['0','1'])

            if st.button('Предсказать'):
                data = list(dt.values())
                data = [int(str) for str in data]
                data = np.array(data).reshape((1, -1))
                pred = model.predict(data)
                st.write(f"Предсказанное значение: {pred[0]:.2f}")
            else:
                pass


@st.cache_data
def load_model(path_to_file):
    with open(path_to_file, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


@st.cache_data
def load_test_data(path_to_file):
    df = pd.read_csv(path_to_file, sep=";", encoding="utf-8")
    y = df["RainTomorrow"]
    X = df.drop(["RainTomorrow"], axis=1)
    dt = dict()
    for str in list(X.columns.values):
        dt[str]=0
    dt_now = datetime.datetime.now()
    dt['Year'] = dt_now.year
    dt['Month'] = dt_now.month
    dt['Day'] =  dt_now.day
    X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X, y, stratify=y, test_size=0.2)
    scaler = StandardScaler()
    X_train_cv = scaler.fit_transform(X_train_cv)
    X_test_cv = scaler.transform(X_test_cv)
    return X_test_cv, y_test_cv, dt


if __name__ == '__main__':
    main()