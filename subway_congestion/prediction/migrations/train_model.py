import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 데이터 로드
data = pd.read_csv('C:\\BigDataProject\\subway_congestion\\data\\filtered_train_time.csv')
print(data.columns)
# '승차'와 '하차' 열 생성
boarding = data[data['구분'] == '승차']
alighting = data[data['구분'] == '하차']

# 필요한 열만 선택하여 다시 합침
boarding = boarding[['날짜', '역번호', '시간대', '인원수']].rename(columns={'인원수': '승차'})
alighting = alighting[['날짜', '역번호', '시간대', '인원수']].rename(columns={'인원수': '하차'})

# '승차'와 '하차' 데이터를 병합
data = pd.merge(boarding, alighting, on=['날짜', '역번호', '시간대'])

# 혼잡도 계산
data['혼잡도'] = data['승차'] - data['하차']

# 사용 가능한 시간대
time_slots = ['06시 이전', '06시-07시', '07시-08시', '08시-09시', '09시-10시', '10시-11시', '11시-12시', 
              '12시-13시', '13시-14시', '14시-15시', '15시-16시', '16시-17시', '17시-18시', '18시-19시', 
              '19시-20시', '20시-21시', '21시-22시', '22시-23시', '23시-24시', '24시 이후']

# 피처와 타겟 설정
features = ['역번호'] + time_slots
data_pivot = data.pivot_table(index=['날짜', '역번호'], columns='시간대', values=['승차', '하차'], fill_value=0)

# 인덱스를 리셋하여 피처 데이터프레임 생성
X = data_pivot.reset_index()
y = data_pivot['혼잡도'].reset_index(drop=True)

# 모델 훈련
model = LinearRegression()
model.fit(X, y)

# 모델 저장
joblib.dump(model, 'congestion_model.pkl')