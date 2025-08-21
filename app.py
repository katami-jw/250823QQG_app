import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

#テキスト表示
st.title("ランダムな都市で巡回セールスマン問題を実施")
st.write("都市数を入力して実行ボタンをクリック")

#都市数を入力
num_cities = st.number_input(`都市数を入力`)

#計算を実行
def city_plot(num_cities,scale=100,seed=None):
  # 都市の座標を生成
  np.random.seed(seed)
  city_xy = np.random.randint(0,scale,(N, 2))

  # 都市間の距離を計算
  x = city_xy[:, 0]
  y = city_xy[:, 1]
  distance_matrix = np.sqrt((x[:, np.newaxis] - x[np.newaxis, :]) ** 2 +(y[:, np.newaxis] - y[np.newaxis, :]) ** 2)

  return city_xy,distance_matrix

N = num_cities

city_xy,distance_matrix = city_plot(N,seed=10)

import amplify

gen = amplify.VariableGenerator()
q = gen.array("Binary", shape=(N + 1, N))
q[N, :] = q[0, :]

objective: amplify.Poly = amplify.einsum("ij,ni,nj->", distance_matrix, q[:-1], q[1:])  # type: ignore

# 最後の行を除いた q の各行のうち一つのみが 1 である制約
row_constraints = amplify.one_hot(q[:-1], axis=1)

# 最後の行を除いた q の各列のうち一つのみが 1 である制約
col_constraints = amplify.one_hot(q[:-1], axis=0)

constraints = row_constraints + col_constraints

constraints *= np.amax(distance_matrix)  # 制約条件の強さを設定
model = objective + constraints

client = amplify.FixstarsClient()
client.token = 'AE/ko4zLSbvzz7WdxZ9ZGv6dAFgTES5gtgz'
client.parameters.timeout = timedelta(milliseconds=1000)  # タイムアウト 1000 ミリ秒

# ソルバーの設定と結果の取得
result = amplify.solve(model, client)
if len(result) == 0:
    raise RuntimeError("At least one of the constraints is not satisfied.")

result.best.objective
q_values = q.evaluate(result.best.values)
route = np.where(np.array(q_values) == 1)[1]

def show_route(route: np.ndarray, distances: np.ndarray, locations: np.ndarray):
    path_length = sum([distances[route[i]][route[i + 1]] for i in range(N)])
    print(f"最短距離: {path_length}")

    x = [i[0] for i in locations]
    y = [i[1] for i in locations]
    plt.figure(figsize=(7, 7))
    plt.xlabel("x")
    plt.ylabel("y")

    for i in range(N):
        r = route[i]
        n = route[i + 1]
        plt.plot([x[r], x[n]], [y[r], y[n]], "b-")
    plt.plot(x, y, "ro")
    plt.show()

    return path_length

path_length = show_route(route, distance_matrix, city_xy)

st.write(path_length)
