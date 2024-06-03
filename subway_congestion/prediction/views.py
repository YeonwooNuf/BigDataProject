from django.shortcuts import render
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 모델 로드
model_path = os.path.join(BASE_DIR, 'prediction', 'C:\\BigDataProject\\subway_congestion\\prediction\\congestion_model.pkl')
model = joblib.load(model_path)

# 데이터 로드
data_path = os.path.join(BASE_DIR, 'data', 'C:\\BigDataProject\\subway_congestion\data\\filtered_train_time.csv')
data = pd.read_csv(data_path)
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
        '방화-하남검단산': ['방화', '개화산', '김포공항', '송정', '마곡', '발산', '우장산', '화곡', '까치산',
        '신정(은행정)', '목동', '오목교(목동운동장앞)', '양평', '영등포구청', '영등포시장', '신길', '여의도',
        '여의나루', '마포', '공덕', '애오개', '충정로(경기대입구)', '서대문', '광화문(세종문화회관)', '종로3가',
        '을지로4가', '동대문역사공원(DDP)', '청구', '신금호', '행당', '왕십리(성동구청)', '마장',
        '답십리', '장한평', '군자(능동)', '아차산(어린이대공원후문)', '광나루(장신대)', '천호(풍납토성)',
        '강동', '길동', '굽은다리(강동구민회관앞)', '명일', '고덕', '상일동', '강일', '미사', '하남풍산',
        '하남시청(덕풍·신장)', '하남검단산'],

        '방화-마천': ['방화', '개화산', '김포공항', '송정', '마곡', '발산', '우장산', '화곡', '까치산',
        '신정(은행정)', '목동', '오목교(목동운동장앞)', '양평', '영등포구청', '영등포시장', '신길', '여의도',
        '여의나루','마포', '공덕', '애오개', '충정로(경기대입구)', '서대문', '광화문(세종문화회관)', '종로3가',
        '을지로4가', '동대문역사공원(DDP)', '청구', '신금호', '행당', '왕십리(성동구청)', '마장',
        '답십리', '장한평', '군자(능동)', '아차산(어린이대공원후문)', '광나루(장신대)', '천호(풍납토성)',
        '강동', '둔춘동', '올림픽공원(한국체대)', '방이', '오금', '개롱', '거여', '마천'],

        '하남검단산-방화': ['하남검단산', '하남시청(덕풍·신장)', '하남풍산', '미사', '강일', '상일동',
        '고덕', '명일', '굽은다리(강동구민회관앞)', '길동', '강동', '천호(풍납토성)', '광나루(장신대)',
        '아차산(어린이대공원후문)', '군자(능동)', '장한평', '답십리', '마장', '왕십리(성동구청)', '행당',
        '신금호', '청구', '동대문역사공원(DDP)', '을지로4가', '종로3가', '광화문(세종문화회관)', '서대문',
        '충정로(경기대입구)', '애오개', '공덕', '마포', '여의나루', '여의도', '신길', '영등포시장',
        '영등포구청', '양평', '오목교(목동운동장앞)', '목동', '신정(은행정)', '까치산', '화곡', '우장산',
        '발산', '마곡', '송정', '김포공항', '개화산', '방화'],

        '마천-방화': ['마천', '거여', '개롱', '오금', '방이', '올림픽공원(한국체대)', '둔춘동', '강동',
        '천호(풍납토성)', '광나루(장신대)', '아차산(어린이대공원후문)', '군자(능동)', '장한평', '답십리',
        '마장', '왕십리(성동구청)', '행당', '신금호', '청구', '동대문역사공원(DDP)', '을지로4가', '종로3가',
        '광화문(세종문화회관)', '서대문', '충정로(경기대입구)', '애오개', '공덕', '마포', '여의나루', '여의도',
        '신길', '영등포시장', '영등포구청', '양평', '오목교(목동운동장앞)', '목동', '신정(은행정)', '까치산',
        '화곡', '우장산', '발산', '마곡', '송정', '김포공항', '개화산', '방화'],

        '하남검단산-마천':['하남검단산', '하남시청(덕풍·신장)', '하남풍산', '미사', '강일', '상일동',
        '고덕', '명일', '굽은다리(강동구민회관앞)', '길동', '강동', '둔춘동', '올림픽공원(한국체대)', '방이',
         '오금', '개롱', '거여', '마천'],

        '마천-하남검단산':['마천', '거여', '개롱', '오금', '방이', '올림픽공원(한국체대)', '둔춘동', '강동',
        '길동', '굽은다리(강동구민회관앞)', '명일', '고덕', '상일동', '강일', '미사', '하남풍산',
        '하남시청(덕풍·신장)', '하남검단산']
    }

    # 출발 및 도착 역이 경로 리스트에 있는지 확인
    if start_station not in stations or end_station not in stations:
        return render(request, 'result.html', {'message': '출발 또는 도착 역이 경로에 없습니다.'})

    route = routes[direction]
    start_index = route.index(start_station)
    end_index = route.index(end_station)
    if start_index < end_index:
        relevant_stations = route[start_index:end_index+1]
    else:
        relevant_stations = route[end_index:start_index+1][::-1]

    # 혼잡도 예측
    혼잡도 = 0
    for station in relevant_stations:
        station_data = data[(data['역명'] == station) & (data['날짜'] == day_of_week)]
        if not station_data.empty:
            station_data_encoded = pd.get_dummies(station_data['시간대'])
            station_data_encoded = station_data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
            prediction = model.predict(station_data_encoded)
            # 혼잡도 계산
            if prediction[0] >= 2000:
                혼잡도 += 100
            elif prediction[0] >= 1000:
                혼잡도 += 50
            elif prediction[0] >= 500:
                혼잡도 += 25
            elif prediction[0] > 0:
                혼잡도 += 10
    
    return render(request, 'result.html', {'congestion': 혼잡도})

