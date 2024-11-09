import empiricaldist
import configparser
import matplotlib.pyplot as plt
import numpy as np
import palmerpenguins
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn.metrics
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats as ss
import session_info

df = sns.load_dataset('tips')

#¿hay valeres null?
print(df.isnull().any())

#demostracion de que este dataset no tiene valores null
print(df.isnull().sum().sum())

#parametros de variables numericas
print(df.describe())

print(df.dtypes)

sns.catplot(
    data=df,
    x='day',
    kind='count',
    palette='pastel'
)
plt.show()

sns.catplot(
    data=df,
    x='time',
    kind='count',
    palette='pastel'
)
plt.show()

df['x']=''

df.pipe(
    lambda dfi: (
        sns.displot(
            data=df,
            x='x',
            hue='sex',
            multiple='fill',
            palette='pastel'
        )
    )
)
plt.show()

df = df.drop(columns='x')

sns.histplot(
    data=df,
    x='tip',
    binwidth=1
)

plt.axvline(
    x=df.tip.mean(),
    color='red',
    linestyle='dashed',
    linewidth=2
)

plt.axvline(
    x=df.tip.quantile(0.75),
    color='green',
    linestyle='dashed',
    linewidth=2
)
plt.axvline(
    x=df.tip.quantile(0.5),
    color='green',
    linestyle='dashed',
    linewidth=2
)
plt.axvline(
    x=df.tip.quantile(0.25),
    color='green',
    linestyle='dashed',
    linewidth=2
)

plt.axvline(
    x=df.tip.max(),
    color='black',
    linestyle='dashed',
    linewidth=2
)

plt.axvline(
    x=df.tip.min(),
    color='black',
    linestyle='dashed',
    linewidth=2
)

plt.show()


#PMFS de tips

sns.histplot(
    data=df,
    x='tip',
    binwidth=1,
    stat='probability'
)

plt.show()

#CMFS de tips

sns.ecdfplot(
    data=df,
    x='tip'
)

plt.show()

sns.ecdfplot(
    data=df,
    x='tip',
    hue='sex'
)
plt.show()

#analisis bivariado

sns.scatterplot(
    data=df,
    x='tip',
    y='total_bill'
)
plt.show()

sns.boxplot(
    data=df,
    y='sex',
    x='tip',
    hue='sex'
)

sns.stripplot(
    data=df,
    y='sex',
    x='tip',
    hue='sex',
    color='.3'
)
plt.show()

sns.violinplot(
    data=df,
    x='sex',
    y='tip'
)

sns.stripplot(
    data=df,
    x='sex',
    y='tip',
    hue='sex',
    color='.3'
)
plt.show()


corr_df = df.drop( columns = [ 'sex', 'smoker', 'day', 'time' ] )

corr_df['sex_numeric'] = df.sex.replace( ['Female', 'Male'], [ 0, 1 ] )
corr_df['day_numeric'] = df.day.replace( [ 'Thur', 'Fri', 'Sat','Sun' ], [ 0, 1, 2, 3 ] )
corr_df['smoker_numeric'] = df.smoker.replace( ['No', 'Yes'], [ 0, 1 ] )

#print(corr_df.dtypes)

sns.heatmap(
    data=corr_df.corr(),
    cmap=sns.diverging_palette(20,230,as_cmap=True),
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    annot=True
)
plt.show()

res_df_1 = scipy.stats.linregress(
    y=df.total_bill,
    x=df.tip
)

res_df_2 = scipy.stats.linregress(
    x=df.tip,
    y=df['size']
)

x_1 = np.array([ df.tip.min(), df.tip.max() ] )
fx_1 = res_df_1.intercept + res_df_1.slope * x_1

sns.scatterplot(
    data=df,
    y='total_bill',
    x='tip'
)

sns.lineplot(
    x=x_1,
    y=fx_1
)

plt.show()

sns.lmplot(
    data=df,
    x='tip',
    y='size',
    height=5
)

plt.show()

#analisis multivariado

#regresion lineal multiple

model_1 = smf.ols(
    formula='tip ~ total_bill',
    data=df
).fit()

print(model_1.summary())

model_2 = smf.ols(
    formula='tip ~ size ',
    data=df
).fit()

print(model_2.summary())

model_3 = smf.ols(
    formula='tip  ~ total_bill + size',
    data=df
).fit()

print(model_3.summary())

model_4 = smf.ols(
    formula='tip  ~ total_bill + size + C(sex)',
    data=df
).fit()

print(model_4.summary())

model_5 = smf.ols(
    formula='tip  ~ total_bill + size + C(day)',
    data=df
).fit()

print(model_5.summary())

model_6 = smf.ols(
    formula='tip  ~ size + total_bill + C(time)',
    data=df
).fit()

#+ C(sex) + C(day)
print(model_6.summary())

models_results = pd.DataFrame(
    dict(
        actual_value = df.tip,
        prediction_model_1 = model_1.predict(),
        prediction_model_2 = model_2.predict(),
        prediction_model_3 = model_3.predict(),
        prediction_model_4 = model_4.predict(),
        prediction_model_5 = model_5.predict(),
        prediction_model_6 = model_6.predict(),
        total_bill = df.total_bill,
        sex = df.sex
    )
)

sns.kdeplot(
    data=models_results,
    x='actual_value'
)
sns.kdeplot(
    data=models_results,
    x='prediction_model_6'
)
plt.show()

#regresion logistica

df['numeric_sex'] = df.sex.replace( [ 'Female', 'Male' ], [ 0, 1 ] )
df['is_smoker'] = df.smoker.replace( ['No', 'Yes'], [ 0, 1 ] )

df = df.astype(
    {
        'numeric_sex':'int64',
        'is_smoker':'int64'
    }
)

model_logic_1 = smf.logit(
    formula='numeric_sex ~ total_bill + tip + C(time)',
    data=df
).fit().summary()

print(model_logic_1)

model_is_smoker = smf.logit(
    formula='is_smoker ~ tip + total_bill + C(sex)+ C(day)+ C(time) + size',
    data=df
).fit()

print(model_is_smoker.params)


is_smoker_df = pd.DataFrame(
    dict(
        actual_smoker = df.is_smoker,
        predicted_values = model_is_smoker.predict().round()
    )
)

print(is_smoker_df.value_counts(['actual_smoker','predicted_values']).reset_index(name='count'))

print(
    sklearn.metrics.confusion_matrix(
        is_smoker_df.actual_smoker,
        is_smoker_df.predicted_values
    )
)

print(
    sklearn.metrics.accuracy_score(
        is_smoker_df.actual_smoker,
        is_smoker_df.predicted_values
    )
)

count_sex_smoker = df.value_counts(['smoker','sex']).reset_index(name='count')

print(count_sex_smoker)