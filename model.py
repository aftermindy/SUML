import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder

df = pd.read_csv('train.csv', delimiter=',', low_memory=False)
df_test = pd.read_csv('test.csv', delimiter=',', low_memory=False)
df = df.iloc[:54000, :]

df.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
df.drop(["unnamed:_0", "id"], axis=1)

df['arrival_delay_in_minutes'] = df['arrival_delay_in_minutes'].fillna(15.178)
df['delay_sum'] = df['departure_delay_in_minutes'] + df['arrival_delay_in_minutes']
df.drop(["arrival_delay_in_minutes", "departure_delay_in_minutes"], axis=1)

columns_to_sum = [
    'inflight_wifi_service', 'departure/arrival_time_convenient', 'ease_of_online_booking',
    'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort',
    'inflight_entertainment', 'on-board_service', 'leg_room_service', 'baggage_handling',
    'checkin_service', 'inflight_service', 'cleanliness'
]

df['survey_sum'] = df[columns_to_sum].sum(axis=1)

df = df.loc[df['flight_distance'] >= 31]
df = df.loc[df['flight_distance'] <= 3748]

df = df.loc[df['delay_sum'] >= 0]
df = df.loc[df['delay_sum'] <= 144]

transform = ColumnTransformer(
    transformers=[
        ('minMax', MinMaxScaler(), ["delay_sum", "survey_sum"]),
        ('standardScaler', StandardScaler(), ["age", "flight_distance"]),
        ('dummy', OneHotEncoder(), ["customer_type", "type_of_travel", "class", "gender"])
    ]
)


def split(dataFrame):
    x_input = dataFrame[["customer_type", "type_of_travel", "class", "gender", "age", "flight_distance", "delay_sum", "survey_sum"]]
    y_input = dataFrame['satisfaction']
    return x_input, y_input


X, y = split(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline(
    [
        ('preprocessor', transform),
        ('classifier', RandomForestClassifier(random_state=42))
    ]
)

pip = pipeline.fit(X_train, y_train)

y_pred = pip.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Macierz pomyłek:")
print(conf_matrix)
print(classification_report(y_test, y_pred))

joblib.dump(pip, 'model.pkl')
