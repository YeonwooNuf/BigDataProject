import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# 데이터 로드
data = pd.read_csv('C:\\BigDataProject\\subway_congestion\\data\\filtered_train_time.csv')

# 시간대별 데이터를 한 열로 펼치는 과정
data = data.melt(id_vars=['날짜', '호선', '역번호', '구분'], 
                 value_vars=['06시 이전', '06시-07시', '07시-08시', '08시-09시', '09시-10시', 
                             '10시-11시', '11시-12시', '12시-13시', '13시-14시', '14시-15시', 
                             '15시-16시', '16시-17시', '17시-18시', '18시-19시', '19시-20시', 
                             '20시-21시', '21시-22시', '22시-23시', '23시-24시', '24시 이후'],
                 var_name='시간대', value_name='인원수')

# 승차와 하차를 나누어 피벗 테이블 생성
boarding = data[data['구분'] == '승차'].pivot_table(index=['날짜', '역번호', '시간대'], values='인원수',aggfunc='sum').reset_index().rename(columns={'인원수': '승차'})
alighting = data[data['구분'] == '하차'].pivot_table(index=['날짜', '역번호', '시간대'], values='인원수', aggfunc='sum').reset_index().rename(columns={'인원수': '하차'})

# 승차와 하차 데이터를 병합
merged_data = pd.merge(boarding, alighting, on=['날짜', '역번호', '시간대'], how='outer').fillna(0)

# 혼잡도 계산
merged_data['혼잡도'] = boarding - alighting

# 피처와 타겟 설정
X = merged_data[['역번호']].join(pd.get_dummies(merged_data['시간대']))  # '시간대'를 원-핫 인코딩
y = merged_data['혼잡도']

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 성능 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 모델 저장
joblib.dump(model, 'C:\\BigDataProject\\subway_congestion\\prediction\\congestion_model.pkl')
