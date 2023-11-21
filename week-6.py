import numpy as np

fruits = np.load('fruits_300.npy')
print(fruits[0, 0, :])

import matplotlib.pyplot as plt
plt.imshow(fruits[0], cmap='gray') # 배경이 검정색으로 출력
plt.imshow(fruits[0], cmap='gray_r') # 배경이 흰색으로 출력
plt.show()

fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()
# fruits에는 100개의 사과, 100개의 바나나, 100개의 파인애플이 저장되어 있음

fruits_2d = fruits.reshape(-1, 100*100)
# reshape으로 두 번째 차원(100)과 세 번재 차원(100)을 합침
print(fruits_2d.shape)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

print(km.labels_)
print(len(km.labels_))

print(np.unique(km.labels_, return_counts=True))

def draw_fruits(arr, ratio=1):
    n = len(arr)
    # n은 샘플의 개수
    # 한 줄에 10개씩 이미지를 그림
    # 샘플 개수를 10으로 나누어 전체 행 개수를 계산
    rows = int(np.ceil(n/10))
    # 행이 1개이면 열의 개수는 샘플 개수, 그렇지 않으면 10개
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n: # n개까지만 그림
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()

draw_fruits(fruits[km.labels_==0])
draw_fruits(fruits[km.labels_==1])
draw_fruits(fruits[km.labels_==2])

print(km.cluster_centers_.reshape(-1, 100, 100).shape)
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
plt.plot(range(2, 7), inertia)
plt.xlabel('x')
plt.ylabel('inertia')
plt.show()