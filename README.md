# Gridiron Guru - NFL Prediction Platform

A comprehensive NFL prediction platform that combines machine learning models with a modern web interface to provide accurate game predictions, team analysis, and user engagement features.

## 🏈 Features

- **Real AI Predictions**: Machine learning models with 61.4% accuracy
- **Live Data**: Real-time NFL data from ESPN API
- **2025 Season**: Complete coverage of all 272 games across 18 weeks
- **Interactive Interface**: Modern React frontend with responsive design
- **Team Analysis**: Detailed team statistics and historical matchups
- **Upset Detection**: Identifies potential upsets with confidence scoring

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/gridiron-guru.git
   cd gridiron-guru
   ```

2. **Backend Setup**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   cd api
   pip install -r requirements.txt
   
   # Start the backend server
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Frontend Setup**
   ```bash
   # In a new terminal
   cd frontend
   npm install
   
   # Start the frontend server
   npm run dev
   ```

4. **Access the Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## 🏗️ Architecture

### Backend (FastAPI)
- **Location**: `/api/`
- **Main File**: `main.py`
- **Port**: 8000
- **ML Models**: Trained ensemble models (Logistic Regression, Random Forest, XGBoost)
- **Data Source**: nfl_data_py library for NFL statistics

### Frontend (React + TypeScript)
- **Location**: `/frontend/`
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Port**: 3000
- **Styling**: Tailwind CSS

### Prediction Engine
- **Location**: `/api/prediction_engine/`
- **Trained Models**: 
  - `ensemble.joblib` - VotingClassifier ensemble
  - `feature_scaler.joblib` - StandardScaler for features
  - Individual model files (.joblib)
- **Training Data**: 2008-2024 NFL seasons
- **Model Accuracy**: 61.4% validation accuracy

## 📊 API Endpoints

- `GET /api/games` - Fetch games with AI predictions
- `GET /api/health` - Health check
- `GET /api/team-stats/{team}` - Team statistics
- `GET /api/historical-matchups/{home}/{away}` - Historical data

## 🔧 Development

### Backend Development
```bash
cd api
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Development
```bash
cd frontend
npm run dev
```

### Testing Predictions
```bash
cd api
python -c "from main import generate_ai_prediction; print(generate_ai_prediction('PHI', 'DAL', 1))"
```

## 📁 Project Structure

```
Gridiron Guru/
├── api/                          # FastAPI backend
│   ├── main.py                   # Main application
│   ├── prediction_engine/        # ML models and prediction logic
│   │   ├── models/trained/       # Trained model files (.joblib)
│   │   ├── prediction_engine.py  # Core prediction logic
│   │   └── feature_engineering.py # Feature creation
│   ├── temporal_pipeline/        # Temporal data processing
│   └── data/                     # NFL data cache (Parquet format)
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── pages/                # Page components
│   │   └── api/                  # API client code
│   └── dist/                     # Built frontend
└── README.md
```

## 🤖 Machine Learning

The platform uses a trained ensemble model combining:
- **Logistic Regression**: Linear baseline model
- **Random Forest**: Tree-based ensemble
- **XGBoost**: Gradient boosting

**Features**: 28 engineered features per game including:
- Team performance metrics
- Historical head-to-head data
- Home field advantage
- Recent form analysis

## 📈 Model Performance

- **Training Data**: 2008-2024 NFL seasons
- **Validation Accuracy**: 61.4%
- **Feature Count**: 28 features per game
- **Prediction Types**: Win probability, score prediction, upset detection

## 🚀 Deployment

### Vercel (Frontend)
The frontend is configured for deployment on Vercel with automatic builds.

### Backend Deployment
The backend can be deployed to any Python hosting service that supports FastAPI.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support, email your-email@example.com or create an issue in the repository.

---

**Gridiron Guru** - Where AI meets NFL predictions! 🏈