# e 증명하기
n = 10 ** 8
e = (1 + 1 / n) ** n
print(e)

# 시그모이드 함수 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
# 시그모이드 함수의 출력이 0.5보다 크면 양성 클래스, 0.5보다 작으면 음성 클래스로 분류.

# 로지스틱 회귀로 도미 빙어 분류하기
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
print(fish.head())
print(pd.unique(fish['Species']))

# 입력과 타깃 데이터 나누기
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

# 훈련, 테스트 데이터셋 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# 훈련과 테스트 세트를 표준화 전처리. 특성의 스케일이 정규화 되지 않으면 여기에 곱해지는 계수 값도 많이 차이나게 됨. 선형 회귀 모델에 규제를 적용할 때 계수 값의 크기가 서로 많이 다르면 공정하게 제어되지 않음.
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 불리언 인덱싱
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# 로지스틱 회귀 모델 훈련하기
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))

print(lr.predict_proba(train_bream_smelt[:5]))
# 첫 번째 열은 음성 클래스에 대한 확률, 두 번째 열은 양성 클래스에 대한 확률을 나타냄.
print(lr.classes_)
# 도미가 음성 클래스, 빙어가 양성 클래스.

# 평가하기
print(lr.coef_, lr.intercept_)
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

# 시그모이드 함수 값 계산
from scipy.special import expit
print(expit(decisions))