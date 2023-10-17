from sklearn import datasets
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()
data = pd.DataFrame(iris.data)
# iris 데이터를 dataframe (2차원 테이블 데이터) 형식으로 바꾼다.
print(data)

data.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
# 2차원 테이블 데이터 열 이름을 지정
feature = data[['Sepal_Length', 'Sepal_Width']]