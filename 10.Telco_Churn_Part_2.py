########################## TELCO CHURN ANALYSIS- FEATURING ENGINEERING  ###########################
# This project includes feature engineering operations, which is the first step in developing a machine learning model.
# It is aimed to create a model that can predict whether customers will churn or not.
# Telco churn data includes information about a fictitious telecom company that provided 7043 California
# home phone and Internet services in the third quarter.
# Shows which install services you've ordered, gone or signed up for.

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


churn = load_data("TelcoCustomerChurn.csv")
df = churn.copy()
df.head()


########################## Structural Changes   ###########################
df.columns = [col.upper() for col in df.columns]

df['TOTALCHARGES'] = pd.to_numeric(df['TOTALCHARGES'], errors = 'coerce')
df["CHURN"] = df["CHURN"].apply(lambda x: 1 if x == "Yes" else 0)


########################## Grab to Columns ###########################
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


########################## Outlier Analysis ###########################
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


########################## Missing Value Analysis ###########################
df.isnull().sum()

df["TOTALCHARGES"].fillna(df["TOTALCHARGES"].median(), inplace = True)

df.isnull().sum()


########################## Feature Engineering  ###########################
df.head()
""" 1 """
df["TENURE"].describe()
df["TENURE"].hist()
bins = [-1, 12, 24, 36, 48, 60, 72]
df["CUSTOMER_AGE_LEVEL"] = pd.cut(df["TENURE"], bins, labels = [1, 2, 3, 4, 5, 6])
df["CUSTOMER_AGE_LEVEL"] = df["CUSTOMER_AGE_LEVEL"].astype("int")

""" 2 """
df["AGE_CHARGES_RATIO"] = df["CUSTOMER_AGE_LEVEL"] / df["MONTHLYCHARGES"]

""" 3 """
SENIOR_CONTRACTS = pd.Series(["no_senior_monthly", "no_senior_yearly", "no_senior_two_year",
                              "senior_monthly", "senior_yearly", "senior_two_year"], dtype = "category")
df["SENIOR_CONTRACTS"] = SENIOR_CONTRACTS

df.loc[(df["SENIORCITIZEN"] == 0) & (df["CONTRACT"] == "Month-to-month"), "SENIOR_CONTRACTS"] = SENIOR_CONTRACTS[0]
df.loc[(df["SENIORCITIZEN"] == 0) & (df["CONTRACT"] == "One year"), "SENIOR_CONTRACTS"] = SENIOR_CONTRACTS[1]
df.loc[(df["SENIORCITIZEN"] == 0) & (df["CONTRACT"] == "Two year"), "SENIOR_CONTRACTS"] = SENIOR_CONTRACTS[2]
df.loc[(df["SENIORCITIZEN"] == 1) & (df["CONTRACT"] == "Month-to-month"), "SENIOR_CONTRACTS"] = SENIOR_CONTRACTS[3]
df.loc[(df["SENIORCITIZEN"] == 1) & (df["CONTRACT"] == "One year"), "SENIOR_CONTRACTS"] = SENIOR_CONTRACTS[4]
df.loc[(df["SENIORCITIZEN"] == 1) & (df["CONTRACT"] == "Two year"), "SENIOR_CONTRACTS"] = SENIOR_CONTRACTS[5]

""" 4 """
GENDER_AGE = pd.Series(["female_1_year", "female_2_year", "female_3_year",
                        "female_4_year", "female_5_year", "female_6_year",
                        "male_1_year", "male_2_year", "male_3_year",
                        "male_4_year", "male_5_year", "male_6_year",
                        ], dtype = "category")
df["GENDER_AGE"] = GENDER_AGE

df.loc[(df["GENDER"] == "Female") & (df["CUSTOMER_AGE_LEVEL"] == 1), "GENDER_AGE"] = GENDER_AGE[0]
df.loc[(df["GENDER"] == "Female") & (df["CUSTOMER_AGE_LEVEL"] == 2), "GENDER_AGE"] = GENDER_AGE[1]
df.loc[(df["GENDER"] == "Female") & (df["CUSTOMER_AGE_LEVEL"] == 3), "GENDER_AGE"] = GENDER_AGE[2]
df.loc[(df["GENDER"] == "Female") & (df["CUSTOMER_AGE_LEVEL"] == 4), "GENDER_AGE"] = GENDER_AGE[3]
df.loc[(df["GENDER"] == "Female") & (df["CUSTOMER_AGE_LEVEL"] == 5), "GENDER_AGE"] = GENDER_AGE[4]
df.loc[(df["GENDER"] == "Female") & (df["CUSTOMER_AGE_LEVEL"] == 6), "GENDER_AGE"] = GENDER_AGE[5]

df.loc[(df["GENDER"] == "Male") & (df["CUSTOMER_AGE_LEVEL"] == 1), "GENDER_AGE"] = GENDER_AGE[6]
df.loc[(df["GENDER"] == "Male") & (df["CUSTOMER_AGE_LEVEL"] == 2), "GENDER_AGE"] = GENDER_AGE[7]
df.loc[(df["GENDER"] == "Male") & (df["CUSTOMER_AGE_LEVEL"] == 3), "GENDER_AGE"] = GENDER_AGE[8]
df.loc[(df["GENDER"] == "Male") & (df["CUSTOMER_AGE_LEVEL"] == 4), "GENDER_AGE"] = GENDER_AGE[9]
df.loc[(df["GENDER"] == "Male") & (df["CUSTOMER_AGE_LEVEL"] == 5), "GENDER_AGE"] = GENDER_AGE[10]
df.loc[(df["GENDER"] == "Male") & (df["CUSTOMER_AGE_LEVEL"] == 6), "GENDER_AGE"] = GENDER_AGE[11]

""" 5 """
service_cols = ['MULTIPLELINES', 'ONLINESECURITY', 'ONLINEBACKUP', 'DEVICEPROTECTION',
                'TECHSUPPORT', 'STREAMINGTV', 'STREAMINGMOVIES', "PHONESERVICE"]

for col in service_cols:
    df[col+"_NEW"] = np.where(df[col] == "Yes", 1, 0)

df["INTERNETSERVICE_NEW"] = np.where(df["INTERNETSERVICE"] == "No", 0, 1)

""" 7 """
df["PHONE_NET"] = np.where((df["PHONESERVICE_NEW"] == 1) & (df["INTERNETSERVICE_NEW"] == 0), 0,
                           np.where((df["PHONESERVICE_NEW"] == 0) & (df["INTERNETSERVICE_NEW"] == 1), 1, 2))

""" 8 """
df["NET_SERVICE_RATE"] = df["STREAMINGMOVIES_NEW"] + df["ONLINESECURITY_NEW"] + df["ONLINEBACKUP_NEW"] +\
                         df["DEVICEPROTECTION_NEW"] + df["TECHSUPPORT_NEW"] + df["STREAMINGTV_NEW"]

df.drop(columns = ['MULTIPLELINES', 'ONLINESECURITY', 'ONLINEBACKUP', 'DEVICEPROTECTION',
                   'TECHSUPPORT', 'STREAMINGTV', 'STREAMINGMOVIES', "PHONESERVICE"], inplace = True)

df.head()


########################## Label Encoding  ###########################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


bin_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype not in ["int64", "float64", "int32"]]

for col in bin_cols:
    label_encoder(df, col)

df.head()
""" 9 """
df["EXTRA_PEOPLE"] = df["PARTNER"] + df["DEPENDENTS"]
df.head()


########################## Rare Encoding  ###########################
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end = "\n\n\n")


new_cat_col = [col for col in df.columns[1:] if col not in num_cols and df[col].dtype != "float64"]
rare_analyser(df, "CHURN", new_cat_col)


########################## One-Hot Encoding  ###########################
ohe_cols = [col for col in df.columns if 14 >= df[col].nunique() > 2 and df[col].dtype in ["object", "category"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
    return dataframe


dff = one_hot_encoder(df, ohe_cols, drop_first = False)
dff.head()
df.shape
dff.shape
dff.columns = [col.upper() for col in dff.columns]
dff.drop(columns = ["CUSTOMERID"], inplace = True)
cat_cols, num_cols, cat_but_car = grab_col_names(dff)
rare_analyser(dff, "CHURN", cat_cols[1:])


########################## Standardization ###########################
ss = StandardScaler()
for col in num_cols:
    dff[col] = ss.fit_transform(df[[col]])
dff.head()


########################## Model ###########################
y = dff["CHURN"]
X = dff.drop(["CHURN"], axis = 1)

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





