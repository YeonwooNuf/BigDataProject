import pandas as pd
import joblib

# Load the saved model
#model = joblib.load('C:\\BigDataProject\\subway_congestion\\prediction\\congestion_model.pkl')

# Load the data
data = pd.read_csv('C:\\BigDataProject\\subway_congestion\\data\\filtered_train_time.csv')

# Define the time slots
time_slots = ['06시 이전', '06시-07시', '07시-08시', '08시-09시', '09시-10시', '10시-11시', '11시-12시', 
              '12시-13시', '13시-14시', '14시-15시', '15시-16시', '16시-17시', '17시-18시', '18시-19시', 
              '19시-20시', '20시-21시', '21시-22시', '22시-23시', '23시-24시', '24시 이후']

# Get a list of unique station names
station_names = data['역명'].unique()

# Function to get congestion for a given station and time slot
def get_congestion(station_name, time_slot):
    # Create a feature vector for the given station and time slot
    station_data = data[(data['역명'] == station_name) & (data['구분'].isin(['승차', '하차']))]
    station_data = station_data.set_index(['날짜', '역명'])[time_slots]
    station_data = station_data.reset_index()
    X = station_data.drop(columns=['날짜', '역명'])
    
    # Get the congestion prediction from the model
    #congestion = model.predict(X)[time_slots.index(time_slot)]
    
    #return congestion

# Example usage
station_name = '신도림'
time_slot = '07시-08시'
congestion_level = get_congestion(station_name, time_slot)
print(f"At {station_name} station during {time_slot}, the congestion level is {congestion_level:.2f}")

X = congestion.drop(columns=['날짜', '역명'])
y = congestion[time_slots].sum(axis=1)  # 각 시간대의 혼잡도 합계를 타겟으로 설정

# 데이터를 훈련 세트와 테스트 세트로 나눔
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 저장
joblib.dump(model, 'C:\\BigDataProject\\subway_congestion\\prediction\\congestion_model.pkl')