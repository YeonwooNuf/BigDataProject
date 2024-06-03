from django.shortcuts import render
import pandas as pd
import joblib

# 모델 로드
model = joblib.load('prediction/congestion_model.pkl')

# 데이터 로드
data = pd.read_csv('data/filtered_train_time.csv')
stations = data['역명'].unique()

def index(request):
    return render(request, 'index.html', {'stations': stations})

def predict(request):
    start_station = request.POST['start_station']
    end_station = request.POST['end_station']
    direction = request.POST['direction']
    time_slot = request.POST['time_slot']
    day_of_week = request.POST['day_of_week']
    
    # 경로 설정
    routes = {
        '방화': ['방화', '개화산', '김포공항', ...],
        '하남검단산': ['하남검단산', '강일', '미사', ...],
        '마천': ['마천', '거여', '개롱', ...]
    }
    
    route = routes[direction]
    start_index = route.index(start_station)
    end_index = route.index(end_station)
    if start_index < end_index:
        relevant_stations = route[start_index:end_index+1]
    else:
        relevant_stations = route[end_index:start_index+1][::-1]

    # 혼잡도 예측
    congestion = 0
    for station in relevant_stations:
        station_data = data[(data['역명'] == station) & (data['날짜'] == day_of_week)]
        features = station_data[['역번호', time_slot]]
        if not features.empty:
            prediction = model.predict(features)
            congestion += prediction[0]
    
    return render(request, 'result.html', {'congestion': congestion})
