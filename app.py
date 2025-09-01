import streamlit as st
import amplify
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

#Pythonの処理まとめ

# 都市間の距離を計算
def city_plot(city_xy,scale=100,seed=None):
    x = city_xy[:, 0]
    y = city_xy[:, 1]
    distance_matrix = np.sqrt((x[:, np.newaxis] - x[np.newaxis, :]) ** 2 +(y[:, np.newaxis] - y[np.newaxis, :]) ** 2)

    return distance_matrix

# 計算をFixstars Amplifyで実行
def tsp(num_city,time_calc):
    gen = amplify.VariableGenerator()
    q = gen.array("Binary", shape=(num_city + 1, num_city))
    q[num_city, :] = q[0, :]

    objective: amplify.Poly = amplify.einsum("ij,ni,nj->", distance_matrix, q[:-1], q[1:])  # type: ignore

    # 最後の行を除いた q の各行のうち一つのみが 1 である制約
    row_constraints = amplify.one_hot(q[:-1], axis=1)

    # 最後の行を除いた q の各列のうち一つのみが 1 である制約
    col_constraints = amplify.one_hot(q[:-1], axis=0)

    constraints = row_constraints + col_constraints

    constraints *= np.amax(distance_matrix)  # 制約条件の強さを設定
    model = objective + constraints

    # ソルバーの設定と結果の取得
    client = amplify.FixstarsClient()
    client.token = 'AE/on8fmXphGRL6EILzXDt6JIsaOVt2SGdU'
    client.parameters.outputs.duplicate = True
    client.parameters.outputs.num_outputs = 0
    client.parameters.timeout = timedelta(milliseconds=time_calc)  # タイムアウト 1000 ミリ秒
    
    result = amplify.solve(model, client)
    if len(result) == 0:
        raise RuntimeError("At least one of the constraints is not satisfied.")
    num_runs = len(result)
    q_values = q.evaluate(result.best.values)
    route = np.where(np.array(q_values) == 1)[1]
    return num_runs,q_values,route

#結果を確認
def show_route(route: np.ndarray, distances: np.ndarray, locations: np.ndarray):
    path_length = sum([distances[route[i]][route[i + 1]] for i in range(N)])

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
    st.pyplot(plt)

    return path_length

#アプリ部分
#テキスト表示
st.title("都市を選択して巡回セールスマン問題を実施")
st.write("巡回する都市を選択してFixstars Amplifyで最短ルートを計算")

#サイドバー設定
with st.sidebar:
    st.write('計算のパラメーター設定')
    time_calc = st.number_input("タイムアウト時間",max_value = 10000, value = 1000, step = 1)

#都市を設定
cities = {
    'A': [23, 56], 'B': [12, 24], 'C': [36, 72], 'D': [9, 53], 'E': [61, 55],
    'F': [47, 19], 'G': [33, 68], 'H': [28, 44], 'I': [71, 32], 'J': [15, 77],
    'K': [52, 26], 'L': [39, 64], 'M': [19, 81], 'N': [48, 23], 'O': [63, 57],
    'P': [26, 49], 'Q': [34, 66], 'R': [54, 21], 'S': [45, 78], 'T': [29, 31],
    'U': [67, 42], 'V': [37, 59], 'W': [41, 73], 'X': [12, 65], 'Y': [55, 36],
    'Z': [22, 48], 'AA': [31, 62], 'AB': [46, 27], 'AC': [14, 75], 'AD': [38, 54],
    'AE': [57, 29], 'AF': [42, 63], 'AG': [64, 18], 'AH': [27, 69], 'AI': [53, 24],
    'AJ': [36, 58], 'AK': [21, 47], 'AL': [68, 35], 'AM': [43, 72], 'AN': [17, 61],
    'AO': [49, 28], 'AP': [32, 67], 'AQ': [59, 33], 'AR': [24, 79], 'AS': [44, 25],
    'AT': [62, 39], 'AU': [35, 71], 'AV': [20, 56], 'AW': [51, 22], 'AX': [30, 74]
}

#都市を選択
col1, col2 = st.columns(2) #横並びレイアウトを作成

with col1:
    selected_cities = st.multiselect('都市を選択',list(cities.keys()),default = list(cities.keys()))
with col2:
    if selected_cities:  # 都市が選択されている場合
        #選択した都市の座標をndarrayにする
        selected_coords = np.array([cities[pref] for pref in selected_cities])
        N = selected_coords.shape[0]  # 都市数

        # 都市の座標をグラフで表示
        plt.figure(figsize=(7, 7))
        plt.plot(selected_coords[:, 0], selected_coords[:, 1], 'o')
        st.pyplot(plt)
    else:  # 都市が選択されていない場合
        selected_coords = np.array([])  # 空のndarrayをセット
        N = 0

        # 空のグラフを表示
        plt.figure(figsize=(7, 7))
        plt.title('No selected')
        st.pyplot(plt)

#計算を実行
button1 = st.button('計算を実行') #ボタンを作成
if button1:
    with st.spinner('計算中...', show_time=True):
        distance_matrix = city_plot(selected_coords,seed=10)   
        num_calc,q_values,route = tsp(N,time_calc)
        path_length = show_route(route, distance_matrix, selected_coords)

    st.write('計算回数:',num_calc)
    st.write('総距離:',path_length)
