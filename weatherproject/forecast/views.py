from django.shortcuts import render
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime, timedelta
import pytz
import os
from pathlib import Path

# Configuration
API_KEY = 'e52a5819f4fa42b766ad805b49a5f848'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = os.path.join(BASE_DIR, 'weather.csv')

def get_current_weather(city):
    """Récupère les données météo actuelles avec coordonnées GPS"""
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 404:
            return None
            
        response.raise_for_status()
        data = response.json()
        
        main_data = data.get('main', {})
        wind_data = data.get('wind', {})
        clouds_data = data.get('clouds', {})
        weather_data = data.get('weather', [{}])[0]
        coord_data = data.get('coord', {})
        
        offset = timedelta(seconds=data.get('timezone', 0))
        utc = pytz.utc
        now = datetime.utcnow().replace(tzinfo=utc) + offset
        
        sunrise = datetime.fromtimestamp(data.get('sys', {}).get('sunrise', 0), utc) + offset
        sunset = datetime.fromtimestamp(data.get('sys', {}).get('sunset', 0), utc) + offset
        is_daytime = sunrise <= now < sunset

        description = weather_data.get('description', 'inconnu')
        if description == 'clear sky':
            description = 'sunny' if is_daytime else 'clear night'

        return {
            'city': data.get('name', 'Inconnu'),
            'current_temp': round(main_data.get('temp', 0)),
            'feels_like': round(main_data.get('feels_like', main_data.get('temp', 0))),
            'temp_min': round(main_data.get('temp_min', 0)),
            'temp_max': round(main_data.get('temp_max', 0)),
            'humidity': main_data.get('humidity', 0),
            'pressure': main_data.get('pressure', 1013),
            'wind_speed': wind_data.get('speed', 0),
            'wind_deg': wind_data.get('deg', 0),
            'clouds': clouds_data.get('all', 0),
            'visibility': data.get('visibility', 10000),
            'description': description,
            'country': data.get('sys', {}).get('country', ''),
            'is_daytime': is_daytime,
            'local_time': now,
            'lat': coord_data.get('lat'),
            'lon': coord_data.get('lon')
        }
        
    except Exception as e:
        print(f"Erreur API météo: {str(e)}")
        return None

def get_air_quality(lat, lon):
    """Récupère les données de qualité de l'air avec les 3 principaux polluants"""
    if not lat or not lon:
        return None
        
    url = f"{BASE_URL}air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        aqi = data['list'][0]['main']['aqi']
        components = data['list'][0]['components']
        
        # Niveaux de qualité de l'air
        aqi_levels = {
            1: {
                'text': "Excellent",
                'description': "L'air est pur et sain.",
                'color': '#4CAF50',
                'icon': 'bi-emoji-smile'
            },
            2: {
                'text': "Bon",
                'description': "Qualité de l'air satisfaisante.",
                'color': '#8BC34A',
                'icon': 'bi-emoji-neutral'
            },
            3: {
                'text': "Modéré", 
                'description': "Qualité acceptable avec quelques polluants.",
                'color': '#FFC107',
                'icon': 'bi-emoji-expressionless'
            },
            4: {
                'text': "Médiocre",
                'description': "Effets possibles sur les personnes sensibles.",
                'color': '#FF9800',
                'icon': 'bi-emoji-frown'
            },
            5: {
                'text': "Mauvais",
                'description': "Risque pour la santé.",
                'color': '#F44336',
                'icon': 'bi-emoji-dizzy'
            }
        }
        
        level_info = aqi_levels.get(aqi, {
            'text': "Inconnu",
            'description': "Données non disponibles",
            'color': '#9E9E9E',
            'icon': 'bi-question-circle'
        })
        
        return {
            'aqi': aqi,
            'text': level_info['text'],
            'description': level_info['description'],
            'color': level_info['color'],
            'icon': level_info['icon'],
            'pollutants': {
                'pm2_5': round(components.get('pm2_5', 0)),
                'pm10': round(components.get('pm10', 0)),
                'no2': round(components.get('no2', 0))
            }
        }
    except Exception as e:
        print(f"Erreur API qualité air: {str(e)}")
        return None

def load_historical_data():
    """Charge les données historiques pour l'entraînement des modèles"""
    try:
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"Fichier introuvable: {CSV_PATH}")
            
        df = pd.read_csv(CSV_PATH)
        
        required_cols = ['MinTemp', 'MaxTemp', 'Temp', 'Humidity', 'Pressure', 'WindGustSpeed', 'RainTomorrow']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
            
        df = df.dropna(subset=required_cols).drop_duplicates()
        df['TempRange'] = df['MaxTemp'] - df['MinTemp']
        df['is_cold'] = (df['Temp'] < 10).astype(int)
        
        return df
        
    except Exception as e:
        print(f"Erreur chargement données: {str(e)}")
        return None

def prepare_models(df):
    """Prépare les modèles de prédiction météo"""
    try:
        features = ['MinTemp', 'MaxTemp', 'Humidity', 'Pressure', 'WindGustSpeed', 'TempRange']
        
        rain_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rain_model.fit(df[features], df['RainTomorrow'])
        
        temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        temp_model.fit(df[features], df['Temp'])
        
        humidity_model = RandomForestRegressor(n_estimators=100, random_state=42)
        humidity_model.fit(df[features], df['Humidity'])
        
        return rain_model, temp_model, humidity_model
        
    except Exception as e:
        print(f"Erreur entraînement: {str(e)}")
        return None, None, None

def weather_view(request):
    """Vue principale pour l'affichage météo"""
    if request.method == 'POST':
        city = request.POST.get('city', '').strip()
        if not city:
            return render(request, 'weather.html', {'error_message': "Veuillez entrer une ville"})

        current = get_current_weather(city)
        if not current:
            return render(request, 'weather.html', {'error_message': "Nom de ville ou pays incorrect. Veuillez vérifier votre saisie."})

        air_quality = get_air_quality(current.get('lat'), current.get('lon'))

        df = load_historical_data()
        if df is None:
            return render(request, 'weather.html', {'error_message': "Erreur de chargement des données historiques"})

        rain_model, temp_model, humidity_model = prepare_models(df)
        if not rain_model:
            return render(request, 'weather.html', {'error_message': "Erreur de préparation des modèles"})

        input_data = {
            'MinTemp': current['temp_min'],
            'MaxTemp': current['temp_max'],
            'Humidity': current['humidity'],
            'Pressure': current['pressure'],
            'WindGustSpeed': current['wind_speed'],
            'TempRange': current['temp_max'] - current['temp_min']
        }
        
        try:
            rain_prob = rain_model.predict_proba(pd.DataFrame([input_data]))[0][1]
            temp_pred = temp_model.predict(pd.DataFrame([input_data]))[0]
            humidity_pred = humidity_model.predict(pd.DataFrame([input_data]))[0]
            
            base_temp = current['current_temp']
            base_humidity = current['humidity']
            
            future_temp = [
                round(base_temp + i*0.3, 1) if temp_pred > base_temp 
                else round(base_temp - i*0.2, 1) 
                for i in range(1, 6)
            ]
            
            future_humidity = [
                max(0, min(100, round(base_humidity + i*(humidity_pred - base_humidity)/5)))
                for i in range(1, 6)
            ]
            
            visibility = current['visibility']
            visibility_display = f"{visibility/1000:.1f} km" if visibility >= 1000 else f"{visibility} m"
            
            now = current['local_time']
            future_times = [(now + timedelta(hours=i)).strftime("%H:%M") for i in range(1, 6)]
            
            context = {
                'location': city,
                'current_temp': current['current_temp'],
                'feels_like': current['feels_like'],
                'humidity': current['humidity'],
                'pressure': current['pressure'],
                'wind': current['wind_speed'],
                'clouds': current['clouds'],
                'visibility': visibility_display,
                'description': current['description'],
                'city': current['city'],
                'country': current['country'],
                'time': now.strftime("%I:%M %p"),
                'date': now.strftime("%B %d, %Y"),
                'is_daytime': current['is_daytime'],
                'time1': future_times[0],
                'time2': future_times[1],
                'time3': future_times[2],
                'time4': future_times[3],
                'time5': future_times[4],
                'temp1': f"{future_temp[0]}",
                'temp2': f"{future_temp[1]}",
                'temp3': f"{future_temp[2]}",
                'temp4': f"{future_temp[3]}",
                'temp5': f"{future_temp[4]}",
                'hum1': f"{future_humidity[0]}",
                'hum2': f"{future_humidity[1]}",
                'hum3': f"{future_humidity[2]}",
                'hum4': f"{future_humidity[3]}",
                'hum5': f"{future_humidity[4]}",
                'rain_prediction': 'Yes' if rain_prob > 0.5 else 'No',
                'rain_probability': f"{rain_prob*100:.1f}%",
                'air_quality': air_quality,
                'MinTemp': current['temp_min'],
                'MaxTemp': current['temp_max']
            }
            
            return render(request, 'weather.html', context)
            
        except Exception as e:
            print(f"Erreur prédiction: {str(e)}")
            return render(request, 'weather.html', {'error_message': "Erreur de prévision météo"})
    
    return render(request, 'weather.html')