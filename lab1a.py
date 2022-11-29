import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import OneHotEncoder

df = pandas.read_csv(
    'C:/Users/oksana.lashchenko/Downloads/СМКММ_lab_1a/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv', )
# print(df)

print('1.Data cleansing', '\n')
# The whole set contains 60 features. missing data inserted into the set looks like a line with the entry “Not Available”.
# You should analyze data and convert number types into float, if value is “Not Available” convert into «not a number»)
df = df.replace('Not Available', np.nan)
print(df.info())

for column in df.columns:
    if (
            'ft' in column or 'kBtu' in column or 'kWh' in column or 'therms' in column or 'gal' in column or 'Score' in column):
        df[column] = df[column].astype(float)
print(df.info())

print('2. Find Missing Data and Outliers')
# a) You need to estimate how much data in columns is missing. If feature has more than 50% gaps, remove it.')
for column in df.columns:
    percentOfEmptyData = df[column].isnull().sum() / df.shape[0]
    if percentOfEmptyData >= 0.5:
        df = df.drop(column, axis=1)
print(df.info())

print('b) getting rid of the outliers', '\n')
for column in df.columns:
    if (
            'Score' in column or 'ft' in column or 'kBtu' in column or 'kWh' in column or 'therms' in column
            or 'gal' in column or 'Metric' in column):
        first_quartile = df[column].describe()['25%']
        third_quartile = df[column].describe()['75%']
        iq = third_quartile - first_quartile
        print(df[column].name)
        column_data = df[(df[column] > (first_quartile - 3 * iq)) & (df[column] < (third_quartile + 3 * iq))]
        print(column_data.info, '\'')
        print('______________________________________________________________________________________________________')

# fig = plt.figure()
first_quartile = df['ENERGY STAR Score'].describe()['25%']
third_quartile = df['ENERGY STAR Score'].describe()['75%']
iq = third_quartile - first_quartile
column_data = df[(df['ENERGY STAR Score'] > (first_quartile - 3 * iq))
                 & (df['ENERGY STAR Score'] < (third_quartile + 3 * iq))]
plt.hist(column_data['ENERGY STAR Score'].dropna(), bins=20, color='blue', edgecolor='black')
plt.title('ENERGY STAR Score')

print('3. Conduct Exploratory Data Analysis')
# а) Finding relationships between various features. Build Density plot (seaborn) between that features
# (Energy Star Scores) by Building Type, Borough. What feature shows better difference between type?
building = df['Largest Property Use Type'].value_counts()
building = list(building[building.values > 100].index)
fig2 = plt.figure()
for property_type in building:
    subset = df[df['Largest Property Use Type'] == property_type]
    seaborn.kdeplot(subset['ENERGY STAR Score'].dropna(), label=property_type, fill=True)
plt.title('Density Plot of ENERGY STAR Score by Property Use Type')
plt.legend()

district = df['Borough'].value_counts()
district = list(district[district.values > 7].index)
fig3 = plt.figure()
for district_type in district:
    subset = df[df['Borough'] == district_type]
    seaborn.kdeplot(subset['ENERGY STAR Score'].dropna(), label=district_type, fill=True)
plt.title('Density Plot of ENERGY STAR Score by Borough')
plt.legend()

print('b) Find the biggest and the lowest correlation between features and (Energy Star Scores)')
correlations = column_data.corr()['ENERGY STAR Score'].sort_values()
print(correlations.head(3), '\n')
print(correlations.tail(4), '\n')
print('______________________________________________________________________________________________________')

print('4. Feature Engineering and Selection')
print('a) One-hot coding of categorical features (district and building type)', '\n')
one_code_enc = OneHotEncoder(handle_unknown='ignore')
transformed = one_code_enc.fit_transform([df['Largest Property Use Type'], df['Borough']])
print(transformed, '\n')

print('b) Taking the logarithm (natural)of numerical data')
for column in df.columns:
    if (
            'ft' in column or 'kBtu' in column or 'kWh' in column or 'therms' in column or 'gal' in column or 'Metric' in column):
        values = df[column].value_counts().sort_values()
        print(values, '\n')
for column in df.columns:
    if (
            'Largest Property Use Type - Gross Floor Area (ft²)' in column or
            'Natural Gas Use (kBtu)' in column or
            'Weather Normalized Site Natural Gas Use (therms)' in column or
            'Electricity Use - Grid Purchase (kBtu)' in column or
            'Weather Normalized Site Electricity (kWh)' in column or
            'Property GFA - Self-Reported (ft²)' in column
    ):
        column = np.log(df[column])
        print(column, '\n')
        print('______________________________________________________________________________________________________')

print('c)Selection')
# You should use correlation coefficient to identify and remove collinear features.
# We will drop one of a pair of features if the correlation coefficient between them is greater than 0.6.

for column in df.columns:
    if (
            'ft' in column or 'kBtu' in column or 'kWh' in column or 'therms' in column or 'gal' in column or 'Metric' in column):
        column_correlations = column_data.corr()[column].sort_values()
        print(column_correlations, 'n')
        print('______________________________________________________________________________________________________')

df_cleared = df.drop(columns=['DOF Gross Floor Area', 'Property GFA - Self-Reported (ft²)', 'Natural Gas Use (kBtu)',
                              'Weather Normalized Site Natural Gas Intensity (therms/ft²)',
                              'Source EUI (kBtu/ft²)',
                              'Weather Normalized Source EUI (kBtu/ft²)', 'Weather Normalized Site EUI (kBtu/ft²)',
                              'Weather Normalized Site Electricity (kWh)',
                              'Electricity Use - Grid Purchase (kBtu)',
                              'Largest Property Use Type - Gross Floor Area (ft²)',
                              'Water Intensity (All Water Sources) (gal/ft²)', 'Total GHG Emissions (Metric Tons CO2e)',
                              'Weather Normalized Site Natural Gas Use (therms)'])
print('CLEARED', '\n')
for column in df_cleared.columns:
    if (
            'ft' in column or 'kBtu' in column or 'kWh' in column or 'therms' in column or 'gal' in column or 'Metric' in column):
        column_correlations2 = df_cleared.corr()[column].sort_values()
        print(column_correlations2, 'n')
        print('______________________________________________________________________________________________________')
print(df_cleared.info())
