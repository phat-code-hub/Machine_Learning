import pandas as pd

groups=['12|3','23|1','13|2']
df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df[groups[0]]=df['Pclass']<=2
df[groups[1]]=df['Pclass']>=2
df[groups[2]]=df['Pclass'] !=2
# same: df[groups[2]]=(df['Pclass'] ==1) | (df['Pclass']==3)
#--------------------------------------------------
def gini(x,y):
    return round(2*(x/y)*(1-x/y),4)
#--------------------------------------------------
def groupInfo(gr1,gr2='Survived',gr3='Sex'):
    tk=df.groupby([gr1,gr2])
    ff,ft,tf,tt=tk[gr3].count()
    gain_info=infomation_Gain(ff,ft,tf,tt)
    print('Infomation gain of group',gr1,'=',gain_info)
    #codes below show detail infos of group
    # left,right=gr1.split('|')
    # print()
    # print('Split by ',gr1,':')
    # print('Class', right,' died: ',ff)
    # print('Class', right' survived: ',ft)
    # print('Class', left,' died: ',tf)
    # print('Class', left,' survived: ',tt)
#--------------------------------------------------
def infomation_Gain(*info):
    e,f,g,h=info
    #All people info
    s_survived=f+h
    s=sum(info) # all people
    #left side info
    a=e+f
    a_survived=f
    #right side info
    b=g+h
    b_survived=h
    #calculate gini impurity
    hs=gini(s_survived,s)
    ha=gini(a_survived,a)
    hb=gini(b_survived,b)
    return round(hs-(a/s)*ha-(b/s)*hb,4)
#--------------------------------------------------
#Main code: loop for 3 groups
for group in groups:
    groupInfo(group)