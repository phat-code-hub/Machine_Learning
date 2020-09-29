

import pandas as pd
df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
#A) Splitting on Age =<30 or not
df['under30']=df['Age'] <=30
u30=df.groupby(['under30','Survived'])
#note: we can change 'Sex' by other feature if we want
print(u30['Sex'].count())
ff,ft,tf,tt=u30['Sex'].count()
# print()
# print('Split by Age 30: ')
# print()
# print('Above 30 did not Survive: ',ff)
# print('Above 30 survived: ', ft)
# print('Under and equal 30 did not survived: ',tf)
# print('Under and equal 30 survived: ',tt)
# #B)Splitting on Sex
# df['male']=df['Sex']=='male'
# genre=df.groupby(['male','Survived'])
# print()
# print('Split by Sex : male and female:')
# print()
# #note : can change 'Age' by other feature
# ff,ft,tf,tt=genre['Age'].count()
# print('Female did not Survive: ',ff)
# print('Female survived: ', ft)
# print('Male did not survived: ',tf)
# print('Male 30 survived: ',tt)

# cl=df.groupby(['Pclass','Survived'])
# print(cl['Sex'].count()) 