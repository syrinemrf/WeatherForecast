# WeatherForecast

WeatherForecast is a Django web application for displaying current weather, air quality, and daily forecasts using the OpenWeatherMap API. The project includes data visualization and a machine learning model for weather prediction.

## Features
- Current weather by city
- Air quality index and main pollutants
- Weekly weather forecast
- Data visualization (charts, tables)
- Machine learning-based weather prediction

## Requirements
- Python 3.10+
- Django 5.2+
- Requests, Pandas, NumPy, scikit-learn, pytz
- OpenWeatherMap API key

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/syrinemrf/WeatherForecast.git
   cd WeatherForecast
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # On Windows
   # Or
   source myenv/bin/activate  # On Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables:
   - Create a `.env` file at the project root:
     ```env
     OPENWEATHER_API_KEY=your_openweather_api_key
     ```
5. Run migrations and collect static files:
   ```bash
   cd weatherproject
   python manage.py migrate
   python manage.py collectstatic
   ```
6. Start the development server:
   ```bash
   python manage.py runserver
   ```

## Deployment
- For production, set `DEBUG = False` and configure `ALLOWED_HOSTS` in `settings.py`.
- Use a production-ready server (Gunicorn, uWSGI) and a web server (Nginx, Apache).
- Store sensitive keys in environment variables or a secure vault.

## Security
- Never commit your `.env` file or secret keys to version control.
- The `.env` file is included in `.gitignore` by default.

## License
This project is licensed under the MIT License.

## Author
- syrinemrf
