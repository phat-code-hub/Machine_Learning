import numpy as np
from  sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#-------------------------------------
#Calculate frequent of repeatable Unicode code
def countPoint(text):
    #Distribute address for Unicode code
    counter=np.zeros(65535) # 2^16
    for i in range(len(text)):
        code_point=ord(text[i])
        if code_point>65535:
            continue
        counter[code_point] +=1
    counter=counter/len(text)
    return counter
#-------------------------------------
#Languages and samples contents are optional
#Note : to incease accuracy with similar ones (like same Latin system group),  
#need to increase the length, complexity of training and test samples  
languages=['Vietnam','English','Japan','Hindi','Russia'] #Optional
train_samples=[
			['Chó giúp con người rất nhiều việc như trông nhà cửa, săn bắt,  \
				và được xem như là loài vật trung thành, tình nghĩa nhất với con người.', 
				'Mèo là những con vật có kỹ năng của thú săn mồi và được biết đến với \
				khả năng săn bắt hàng nghìn loại sinh vật để làm thức ăn.', \
				'Lợn không có tuyến bài tiết mồ hôi, vì thế chúng phải tìm các nơi \
				 râm mát hay ẩm ướt (các nguồn nước, vũng bùn v.v) để tránh bị quá nóng \
				 trong điều kiện thời tiết nóng.'],
			['Dogs help people with a lot of things like housework, \
				hunting, and are regarded as the most loyal and affectionate animals to humans.',
				'Cats are skilled animals of predators and are known \
				for their ability to hunt thousands of creatures for food.',
				'Pigs do not have a sweat gland, so they must look for shady \
				or damp places (water sources, puddles, etc.) to avoid overheating in hot weather.'],
			['犬は家事や狩猟などの多くのことで人々を助け、人間にとって最も忠実で愛情のこもった動物とみなされています。',
			 '猫は捕食者の熟練した動物であり、食べ物を求めて何千もの生き物を狩る能力で知られています。',
			 '豚には汗腺がないため、暑い季節に過熱しないように、日陰や湿った場所（水源、水たまりなど）を探す必要があります'],
			['कुत्ते लोगों को गृहकार्य, शिकार जैसी कई चीजों में मदद करते हैं और उन्हें मनुष्यों के लिए सबसे वफादार और स्नेही जानवर माना जाता है।',
				'बिल्लियों शिकारियों के कुशल जानवर हैं और भोजन के लिए हजारों जीवों का शिकार करने की उनकी क्षमता के लिए जाने जाते हैं।',
				'सूअरों के पास पसीने की ग्रंथि नहीं होती है, इसलिए उन्हें गर्म मौसम में गर्मी से बचने के \
				 लिए छायादार या नम स्थान (पानी के स्रोत, पोखर आदि) की तलाश करनी चाहिए।'],
             ['Собаки помогают людям во многих вещах, таких как работа по дому, охота, и \
              считаются самыми верными и ласковыми животными для людей',
              'Кошки являются искусными животными хищников и известны своей способностью \
                  охотиться на тысячи существ в поисках пищи.',
             'У свиней нет потовых желез, поэтому они должны искать тенистые или влажные места \
                 (источники воды, лужи и т. Д.), Чтобы избежать перегрева в жаркую погоду'
                 ]
             
		]

train_dict=dict(zip(languages,train_samples))
X_train=[]
y_train=[]
X_test=[]
for lan,contents in train_dict.items():
    for content in contents:
        y_train.append(lan)
        X_train.append(countPoint(content))
#Machine training
model=GaussianNB()
model.fit(X_train,y_train)
#Test sample  (Optional)
test_samples=['My hometown is England.','今日の天気は雨です。','मेरा गृहनगर वियतनाम है।','Мой родной город - Вьетнам.','私の故郷は日本です。','Quê tôi là việt nam']
for test in test_samples:
    X_test.append(countPoint(test))
y_test=['English','Japan','Hindi','Russia','Japan','Vietnam']

#Predict sample
y_pred=model.predict(X_test)
print('Test language: ',y_test)
print('Predicted result:' ,y_pred)
print('Accuracy: ',accuracy_score(y_test,y_pred))


       