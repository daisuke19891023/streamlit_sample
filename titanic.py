import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, accuracy_score)
st.title('Titanic Estimater')


@st.cache
def load_data(file: str):
    data = pd.read_csv(file)
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Load 10,000 rows of data into the dataframe.
data = load_data("train.csv")
# データのimport
data_load_state.text("Done! (using st.cache)")

if st.sidebar.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.balloons()
    st.write(data)

if st.sidebar.checkbox('Show chart data'):
    # bad case
    # st.line_chart(data)
    st.line_chart(data["Fare"])

    sns.boxplot(x=data.Survived, y=data.Age)
    st.pyplot()

# if st.button('Train'):


# preprosessing
def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * null_val/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
        columns={0: '欠損数', 1: '%'})
    return kesson_table_ren_columns


if st.sidebar.checkbox('Show Missing Data'):
    st.write(kesson_table(data))


# Machine Learning
def padding_data(df):
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    return df


def convertToDummies(df_org):
    # train
    df = pd.DataFrame(df_org)
    df['Sex'] = pd.get_dummies(df['Sex'], drop_first=True)
    df_Embarked = pd.get_dummies(
        df['Embarked'], drop_first=True, prefix='Embarked', prefix_sep='_')
    df = pd.concat([df, df_Embarked], axis=1)
    return df


@st.cache
def preprocessing(df):
    # result = padding_data(df)
    result = convertToDummies(df)
    return result


@st.cache
def train(df):
    # データセットを生成する
    features = ["Pclass", "Sex", "Age", "Fare",
                "Embarked_Q", "Embarked_S", "SibSp", "Parch"]
    X = df[features]
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    lgbm_params = {
        # 分類
        'objective': 'binary',
        'metric': 'binary_logloss',
        "verbosity": -1}

    # 上記のパラメータでモデルを学習する
    model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)

    # テストデータを予測する
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    st.text('AUC='+str(auc(fpr, tpr)))
    st.text('accuracy='+str(accuracy_score(y_test, np.round(y_pred))))

    importance = pd.DataFrame(
        model.feature_importance(), index=features, columns=['importance'])
    st.write(importance)
    # st.write(model.feature_importance())
    return model


if st.sidebar.button('train'):
    process_state = st.text('Preprocessing...')

    df = preprocessing(data)

    process_state.text("Done! (using st.cache)")

    st.write(df)
    model = train(df)
