import pandas

titanic = pandas.read_csv('C:/Users/oksana.lashchenko/Documents/titanic.csv', )

print('1. How many men and women were on the ship?')
sex = titanic['Sex'].value_counts()
print(sex, '\n')

print('2. How many passengers survived?')
survived = titanic['Survived'].value_counts()
total = titanic['PassengerId'].value_counts()
print(survived[1])
print(round((survived[1]/len(total)) * 100, 2), '%', '\n')

print('3. What is the share of first class passengers among all passengers?')
pClass = titanic['Pclass'].value_counts()
total = titanic['PassengerId'].value_counts()
print(pClass[1])
print(round((pClass[1]/len(total)) * 100, 2), '%',  '\n')

print('4. How old were the passengers? Calculate the average and median age of the passengers.')
ageMedian = titanic['Age'].median()
ageAverage = titanic['Age'].mean()
print('average = ', round(ageAverage, 2))
print('median = ', round(ageMedian, 2), '\n')

print('5. Does the number of siblings correlate with the number of parents / children?')
print('Calculate the Pearson correlation between the SibSp and Parch features.')
correlation = titanic["SibSp"].corr(titanic["Parch"])
print('Pearson correlation = ', round(correlation, 2), '\n')

print('6. What is the most popular female name on a ship?')
women = titanic[titanic["Sex"] == "female"]
womenNames = women["Name"].str.split(",").str.get(1)

miss = womenNames[womenNames.str.contains('Miss. ')].str.split('.').str[1]
missPured = miss.str.split(' ').str[1].value_counts()
print('Top3 popular names for Miss')
print(missPured[:3], '\n')

mrs = womenNames[womenNames.str.contains('Mrs.')].str.split('(').str[1]
mrsPured2 = mrs.str.split(' ').str[0].value_counts()
print('Top3 popular names for Mrs')
print(mrsPured2[:3], '\n')

print('Top 2 poplular names')
print('Anna', missPured[0] + mrsPured2[1])
print('Mary', missPured[1] + mrsPured2[2])

# other = []
# other.append(womenNames[womenNames.str.contains('Dr.')])
# other.append(womenNames[womenNames.str.contains('Countess')])
# other.append(womenNames[womenNames.str.contains('Lady')])
print(other)




