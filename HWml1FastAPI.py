from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import List
import pandas as pd
import joblib
import re
import io
import csv

app = FastAPI()

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

NUMERIC_COLUMNS = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']

class CarItem(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class CarItems(BaseModel):
    objects: List[CarItem]

def preprocess_data(data: dict) -> dict:
    processed_data = data.copy()
    # обрабатываем входные данные как в треине
    # получем бренд и удаляем name
    processed_data['brand'] = processed_data['name'].split()[0]
    del processed_data['name']

    # удаляем torque
    del processed_data['torque']

    # убираем ед. изм. в mileage
    processed_data['mileage'] = float(''.join(re.findall(r'-?\d*\.?\d*', processed_data['mileage'])))

    # убираем ед. изм. в engine
    processed_data['engine'] = float(''.join(re.findall(r'-?\d*\.?\d*', processed_data['engine'])))

    # убираем ед. изм. в max_power
    processed_data['max_power'] = float(''.join(re.findall(r'-?\d*\.?\d*', processed_data['max_power'])))

    return processed_data

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # обрабатываем входные данные как в треине
    numeric_features = df[NUMERIC_COLUMNS].copy()
    scaled_features = scaler.transform(numeric_features)
    df[NUMERIC_COLUMNS] = scaled_features

    # добавляем категориальные
    categorical_columns = ['brand', 'fuel', 'seller_type', 'transmission', 'owner']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    # делаем колонки как треине
    expected_features = model.feature_names_in_
    for feature in expected_features:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0

    df_encoded = df_encoded[expected_features]
    return df_encoded

@app.post("/predict_item")
def predict_item(item: CarItem) -> float:
    try:
        data_dict = item.dict()
        processed_data = preprocess_data(data_dict)
        df = pd.DataFrame([processed_data])
        df_encoded = create_features(df)
        prediction = model.predict(df_encoded)[0]
        return max(0, float(prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")

@app.post("/predict_items")
def predict_items(items: List[CarItem]) -> List[float]:
    try:
        processed_items = [preprocess_data(item.dict()) for item in items]
        df = pd.DataFrame(processed_items)
        df_encoded = create_features(df)
        predictions = model.predict(df_encoded)
        return [max(0, float(p)) for p in predictions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        # читаем CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # проверяем наличие необходимых колонок
        required_columns = ['name', 'year', 'selling_price', 'km_driven', 'fuel',
                          'seller_type', 'transmission', 'owner', 'mileage',
                          'engine', 'max_power', 'torque', 'seats']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют необходимые колонки: {missing_columns}")

        # Обрабатываем каждую строку
        processed_rows = []
        for _, row in df.iterrows():
            processed_data = preprocess_data(row.to_dict())
            processed_rows.append(processed_data)

        # создаем df и получаем предикт
        df_processed = pd.DataFrame(processed_rows)
        df_encoded = create_features(df_processed)
        predictions = model.predict(df_encoded)

        # Добавляем предсказания
        df['predicted_price'] = predictions
        output = io.StringIO()
        df.to_csv(output, index=False)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                'Content-Disposition': f'attachment; filename="predictions.csv"'
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке CSV: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
