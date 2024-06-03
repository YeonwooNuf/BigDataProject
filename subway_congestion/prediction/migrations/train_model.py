import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 데이터 로드
data = pd.read_csv('C:\\GitHub\\BigDataProject\\filtered_train_time.csv')

# 시간대 열들
time_slots = ['06시 이전', '06시-07시', '07시-08시', '08시-09시', '09시-10시', '10시-11시', '11시-12시', 
              '12시-13시', '13시-14시', '14시-15시', '15시-16시', '16시-17시', '17시-18시', '18시-19시', 
              '19시-20시', '20시-21시', '21시-22시', '22시-23시', '23시-24시', '24시 이후']

# 승차, 하차 데이터 분리
boarding = data[data['구분'] == '승차'].set_index(['날짜', '역번호', '역명'])[time_slots]
alighting = data[data['구분'] == '하차'].set_index(['날짜', '역번호', '역명'])[time_slots]

# 혼잡도 계산 (승차 - 하차)
congestion = boarding - alighting

# 인덱스를 리셋하여 피처 데이터프레임 생성
congestion = congestion.reset_index()

# 피처와 타겟 설정
X = congestion.drop(columns=['날짜', '역명'])
y = congestion[time_slots].sum(axis=1)  # 각 시간대의 혼잡도 합계를 타겟으로 설정

# 모델 훈련
model = LinearRegression()
model.fit(X, y)

# 모델 저장
joblib.dump(model, 'C:\\GitHub\\BigDataProject\\subway_congestion\\prediction\\congestion_model.pkl')