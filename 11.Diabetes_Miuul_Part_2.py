########################## DIABETES -- FEATURING ENGINEERING   ###########################
# This project includes feature engineering operations, which is the first step in developing a machine learning model.
# It is aimed to establish a model that can predict whether people have diabetes
# when their characteristics are specified.
# The dataset is part of the large dataset held at the National Institutes of Diabetes-Digestive-Kidney Diseases
# in the USA.
# Data used for diabetes research on Pima Indian women aged 21 and over living in Phoenix,
# the 5th largest city in the State of Arizona in the USA.
# The target variable is specified as "outcome"; 1 indicates positive diabetes test result, 0 indicates negative.

########################## Importing Library and Settings  ###########################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

path = "C:\\Users\\hseym\\OneDrive\\Masaüstü\\Miuul\\datasets"
os.chdir(path)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


########################## Loading  The Date  ###########################
def load_data(dataname):
    return pd.read_csv(dataname)


diabetes = load_data("diabetes_miuul.csv")
df = diabetes.copy()
df.head()

########################## Grab to Columns ###########################
df.columns = [col.upper() for col in df.columns]


def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


########################## Outlier Analysis  ###########################
df.isnull().sum()

# Logically, these columns cannot take the value 0.
df[['GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI']] = df[
    ['GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI']].replace(0, np.NaN, )

df.isnull().sum()


def outlier_thresholds(dataframe, col_name, q1 = 0.10, q3 = 0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False


for col in num_cols:
    print(check_outlier(df, col))


def grab_outliers(dataframe, col_name, index = False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


for col in num_cols:
    grab_outliers(df, col)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(check_outlier(df, col))


########################## Missing Value Analysis ###########################
def missing_values_table(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis = 1, keys = ['n_miss', 'ratio'])
    print(missing_df, end = "\n")

    if na_name:
        return na_columns


na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end = "\n\n\n")


missing_vs_target(df, "OUTCOME", na_cols)


# This is a function that gets the median of the columns with respect to the target variable.
def median_target(dataframe, col):
    temp = dataframe[dataframe[col].notnull()]
    temp = temp[[col, 'OUTCOME']].groupby(['OUTCOME'])[[col]].median().reset_index()

    return temp


# Variables with few missing values
col_ind_var = df.columns.drop(["OUTCOME", "INSULIN", "SKINTHICKNESS"])

for col in col_ind_var:
    df.loc[(df["OUTCOME"] == 0) & (df[col].isnull()), col] = median_target(df, col)[col][0]
    df.loc[(df["OUTCOME"] == 1) & (df[col].isnull()), col] = median_target(df, col)[col][1]

df.isnull().sum()

# Variables with many missing values
col_too_nulls = ["INSULIN", "SKINTHICKNESS"]

def too_null_filling(dataframe, col_list):
    """

    This is a function that fills variables with many missing values
    with the median of that variable relative to the target and a random noise.

    """
    for col in col_list:
        if col == "INSULIN":
            dataframe.loc[(dataframe["OUTCOME"] == 0) & (dataframe[col].isnull()), col] = \
                median_target(dataframe, col)[col][0] + np.random.normal(scale = 3.0, size = len(
                    df[(dataframe["OUTCOME"] == 0) & (dataframe[col].isnull())]))

            dataframe.loc[(dataframe["OUTCOME"] == 1) & (dataframe[col].isnull()), col] = \
                median_target(dataframe, col)[col][1] + np.random.normal(scale = 3.0, size = len(
                    df[(dataframe["OUTCOME"] == 1) & (dataframe[col].isnull())]))
        else:
            dataframe.loc[(dataframe["OUTCOME"] == 0) & (dataframe[col].isnull()), col] = \
                median_target(dataframe, col)[col][0] + np.random.normal(scale = 0.5, size = len(
                    df[(dataframe["OUTCOME"] == 0) & (dataframe[col].isnull())]))

            dataframe.loc[(dataframe["OUTCOME"] == 1) & (dataframe[col].isnull()), col] = \
                median_target(dataframe, col)[col][1] + np.random.normal(scale = 0.5, size = len(
                    df[(dataframe["OUTCOME"] == 1) & (dataframe[col].isnull())]))


too_null_filling(df, col_too_nulls)

df.isnull().sum()

########################## Feature Engineering  ###########################
NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")

df["NEW_BMI"] = np.where(df["BMI"] > 39.9, "Obesity 3",
                         np.where(df["BMI"] > 34.9, "Obesity 2",
                                  np.where(df["BMI"] > 29.9, "Obesity 1",
                                           np.where(df["BMI"] > 24.9, "Overweight",
                                                    np.where(df["BMI"] > 18.5, "Normal", "Underweight")))))

df.head()

NewGlucose = pd.Series(["Normal", "Pre_Diabetes", "Diabetes"], dtype = "category")
df["NEW_GLUCOSE"] = np.where(df["GLUCOSE"] > 200, "Diabetes",
                             np.where(df["GLUCOSE"] > 140, "Pre_Diabetes", "Normal"))

df.head()

NewBloodPressure = pd.Series(["Hypotension", "Normal", "Pre_hypertension",
                              "Hypertension_1", "Hypertension_2", "Hypertensive_Crisis"], dtype = "category")

df["NEW_BLOODPRESSURE"] = np.where(df["BLOODPRESSURE"] >= 100, "Hypertension_2",
                                   np.where(df["BLOODPRESSURE"] >= 90, "Hypertension_1",
                                            np.where(df["BLOODPRESSURE"] >= 80, "Pre_hypertension",
                                                     np.where(df["BLOODPRESSURE"] >= 60, "Normal", "Hypotension"))))

df.head()

df["NEW_INSULIN_SCORE"] = np.where((df["INSULIN"] >= 16) & (df["INSULIN"] <= 166), "Normal", "Abnormal")

df.head()

bins = [20, 25, 30, 40, 50, df["AGE"].max()]
df["AGE_SEGMENT"] = pd.cut(df["AGE"], bins, labels = ["Young", "Young+", "Middle", "Older-", "Older"])
df["NEW_AGE_SEGMENT"] = pd.qcut(df["AGE"], 5)


bins_2 = [0, 25, 30, 35, 40, df["SKINTHICKNESS"].max()]
df["SKINTHICKNESS_LEVEL"] = pd.cut(df["SKINTHICKNESS"], bins_2, labels = ["E", "D", "C", "B", "A"])
df["NEW_SKINTHICKNESS_LEVEL"] = pd.qcut(df["SKINTHICKNESS"], 5)


df["PREGNANCIES_AGE_RATE"] = df["PREGNANCIES"] / df["AGE"]
df.head()


########################## Label Encoding  ###########################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


bin_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype not in ["int64", "float64", "int32"]]

for col in bin_cols:
    label_encoder(df, col)


########################## Rare Encoding  ###########################
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end = "\n\n\n")


new_cat_col = [col for col in df.columns if col not in num_cols and df[col].dtype not in ["float64", "int64"]]
rare_analyser(df, "OUTCOME", new_cat_col)


########################## One-Hot Encoding  ###########################
ohe_cols = [col for col in df.columns if 14 >= df[col].nunique() > 2 and df[col].dtype in ["object", "category"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
    return dataframe


dff = one_hot_encoder(df, ohe_cols)
dff.head()
df.shape
dff.shape
dff.columns = [col.upper() for col in dff.columns]
cat_cols, num_cols, cat_but_car = grab_col_names(dff)
rare_analyser(dff, "OUTCOME", cat_cols[1:])
useless_cols = [col for col in dff.columns if dff[col].nunique() == 2 and
                (dff[col].value_counts() / len(dff) < 0.01).any(axis=None)]
dff.drop(useless_cols, axis = 1, inplace = True)


########################## Standardization ###########################
ss = StandardScaler()
for col in num_cols:
    dff[col] = ss.fit_transform(dff[[col]])
dff.head()


########################## Model ###########################
y = dff["OUTCOME"]
X = dff.drop(["OUTCOME"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 17)

rf_model = RandomForestClassifier(random_state = 46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


########################## Feature Importances ###########################
def plot_importance(model, features, num = len(X), save = False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize = (10, 10))
    sns.set(font_scale = 1)
    sns.barplot(x = "Value", y = "Feature", data = feature_imp.sort_values(by = "Value",
                                                                           ascending = False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)
