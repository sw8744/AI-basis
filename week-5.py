from sklearn import datasets
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()
data = pd.DataFrame(iris.data)
# iris 데이터를 dataframe (2차원 테이블 데이터) 형식으로 바꾼다.
print(data)

data.columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
# 2차원 테이블 데이터 열 이름을 지정
feature = data[['Sepal length', 'Sepal width']]

model = KMeans(n_clusters=3, algorithm='auto')
# 모델은 kmeans 알고리즘을 사용하며, 클러스터는 3개(iris 데이터가 3종류의 꼴의 정보를 담고 있음), algorithm='auto'는 기본 값으로 상황에 따라 'full'이나 'elkan' 옵션으로 적용하며 이는 데이터에 따라 다르다.
model.fit(feature) # feature 지정
predict = pd.DataFrame(model.predict(feature))
# feature 값을 토대로 predict 값을 계산하고 이를 dataframe(2차원 테이블 데이터)로 바꾸어 준다.
predict.columns = ['predict']
# dataframe의 열 이름을 predict로 한다.

r = pd.concat([feature, predict], axis=1)
# axis = 0이면 데이터를 위 + 아래로 합치고, 1이면 왼쪽 + 오른쪽으로 합친다.
print(r)

plt.scatter(r['Sepal length'], r['Sepal width'], c=r['predict'], alpha=0.5)

centers = pd.DataFrame(model.cluster_centers_, columns=['Sepal length', 'Sepal width'])
center_x = centers['Sepal width']
center_y = centers['Sepal width']
plt.scatter(center_x, center_y, s=50, marker='D', c='r')