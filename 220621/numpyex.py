# 01. numpy 예제: Broadcasting 예제

import numpy as np
from numpy.random import rand

"""
흑백 이미지: 2차 텐서
RGB 이미지: 3차 텐서(=흑백이미지 + RGB 채널)
RGB 동영상: 4차 텐서(=RGB 이미지 + 시퀀스)

(3, 3) vector1이 있다고 했을 때, (1, 3) vector2와 연산을 하려면 두 dim이 같아야한다. 따라서 vector2를 복제하여 (3, 3)으로 broadcasting 한다.
=> (m, n), (m, 1)에선 1을 n으로 맞춰줄 것이다.

KNN classification
여러 데이터셋이 그래프 상에 있다고 할 때, 특정 좌표 4개를 그래프 상에 그려보자. 이 좌표들이 각각 데이터셋에 분류되려면,
KNN에 의해 각 좌표 4개에 대하여 모든 데이터셋의 데이터들과의 거리를 계산한 후, 이 거리에 따라 특정 데이터셋으로 분류될 것이다.
=> 비효율적. 심지어 for문이라 더 비효율적임. for문 없애야 함.
=> Q. 어떻게 더 효율적으로 구현할 수 있을까?
ex) K means clustering
어떻게 clustering 할 것인가?
단순히 분류하면 모든 데이터셋들의 데이터간에 값을 비교연산하며 m*n... 등으로 비효율적임

"""

# A = np.random.normal(0, 1, size=(3,4))
# print(A.ndim)

# 1차 텐서 두 개를 100개, 200개로 다음과 같이 데이터셋이 주어졌다고 해보자
A = rand(100)
B = rand(200)
print(A.shape, B.shape)
# 우린 shape 값을 다음과 같이 얻을 수 있을 것이다
N1 = A.shape[0]
N2 = B.shape[0]

# 이를 바탕으로, reshape
A = A.reshape(N1, 1)
B = B.reshape(1, N2)
print(A.shape, B.shape)

# <<<< vectorization >>>>
# 이를 broadcasting하면 (N1, N2), (N2, N1)이 될 것이다. 어떻게? 합친다.
# 우리는 이를 fully-connected-operation이라고 부를 수도 있을 것이다 => (N1, N2)
S = A+B
# 한편, 이렇게 만들어 볼 수도 있을 것이다. euclidean distance와 닮아있다. 여기에 그냥 루트 씌우면 됨.
S = (A-B)**2
# S[i][j] = (A[i]-B[j]) ** 2. 우리 이제 for문 없이 거리 계산할 수 있음.


# 이러한 원리는 벡터로 확장될 수 있다. 즉, 차원을 늘려도 이러한 원리는 변하지 않는다.
A = rand(100, 2)
B = rand(200, 2)

# 이와 같이 차원을 하나 늘려도 (N1, 1, ...), (1, N2, ...)인 건 달라지지 않는다.
A = A.reshape(N1, 1, 2)
B = B.reshape(1, N2, 2)
S = A + B       # (N1, N2, 2)
# 위와 마찬가지로 다음과 같이 만들어볼 수 있을 것이다. euclidean distance.
S = (A - B)**2      # S[i][j] = ((x1_i - x2_i)**2), ((y1_i - y2_i)**2)

distances = np.sum(S, axis=-1)      # 마지막 차원을 기준으로 더한다. (N1, N2)로 return.

# 이를 더 확장하면, 즉 S에 대한 식을 바꿔주면, 두 벡터 간의 모든 연산에 대응시킬 수 있다. 차원도 확장 가능하다.
"""
이와 같은 경우에도 마찬가지로 일반화가 적용된다.
(m, _) => (m, 1, 1, _)
(n, _) => (n, 1, 1, _)
(l, _) => (1, 1, l, _)
=> (m, n, l, _)
이거 for문으로 돌리면 3중 for문임. m*n*l만큼 수행함. 근데 이와 같이 vectorization하면? 훨씬 빠름.
"""


