# Gridiron Guru 🏈

The ultimate NFL prediction platform powered by artificial intelligence and comprehensive data analysis. Make smarter picks, understand the game deeper, and dominate your fantasy leagues.

## 🚀 Features

- **AI-Powered Predictions**: Get intelligent game predictions backed by comprehensive NFL data analysis
- **Rich NFL Data**: Access detailed team statistics, player performance, and historical game data
- **Advanced Analytics**: Deep dive into team performance metrics and trend analysis
- **Weekly Predictions**: Make predictions for every game with confidence scoring
- **Real-time Updates**: Stay current with live game data and performance statistics

## 🏗️ Architecture

This project has been completely rebuilt with a modern, scalable architecture:

- **Frontend**: React 19 + TypeScript + Tailwind CSS
- **Backend**: Python FastAPI with nfl_data_py integration
- **Deployment**: Vercel (Frontend + API Functions)
- **Data Sources**: nflfastR, nfldata, DynastyProcess, Draft Scout

## 📁 Project Structure

```
Gridiron Guru/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/          # Page components
│   │   ├── api/            # API service layer
│   │   └── App.tsx         # Main app component
│   ├── package.json        # Frontend dependencies
│   └── vite.config.ts      # Vite configuration
├── api/                     # Python FastAPI backend
│   ├── main.py             # Main FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── env.example         # Environment variables template
├── vercel.json             # Vercel deployment configuration
└── README.md               # This file
```

## 🛠️ Setup Instructions

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- Vercel CLI
- OpenAI API key

### 1. Install Dependencies

#### Frontend
```bash
cd frontend
npm install
```

#### Backend
```bash
cd api
pip install -r requirements.txt
```

### 2. Environment Configuration

#### Backend Environment
```bash
cd api
cp env.example .env
```

Edit `.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=your_openai_api_key_here
CURRENT_SEASON=2024
```

#### Frontend Environment
```bash
cd frontend
```

Create `.env.local`:
```env
VITE_API_URL=http://localhost:8000
```

### 3. Local Development

#### Start Backend
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Start Frontend
```bash
cd frontend
npm run dev
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 4. Vercel Deployment

#### Install Vercel CLI
```bash
npm i -g vercel
```

#### Deploy to Vercel
```bash
vercel
```

Follow the prompts to:
1. Link to your Vercel account
2. Set up the project
3. Configure environment variables

#### Environment Variables in Vercel
Set these in your Vercel dashboard:
- `OPENAI_API_KEY`: Your OpenAI API key
- `CURRENT_SEASON`: Current NFL season (e.g., 2024)

## 🔌 API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /api/teams` - Get all NFL teams
- `GET /api/teams/{team}/stats` - Get team statistics
- `GET /api/games` - Get games for season/week
- `GET /api/players` - Get player statistics
- `GET /api/standings` - Get current standings
- `POST /api/predict` - Generate AI prediction

### Data Sources
The API integrates with multiple NFL data sources:
- **nflfastR**: Play-by-play data and advanced metrics
- **nfldata**: Comprehensive team and player statistics
- **DynastyProcess**: Fantasy football insights
- **Draft Scout**: Draft analysis and combine data

## 🎯 Key Features

### AI Predictions
- User submits game prediction with confidence level
- AI analyzes historical data, team stats, and trends
- Returns detailed prediction with reasoning
- Confidence scoring and performance tracking

### Data Integration
- Real-time NFL data from multiple sources
- Historical performance analysis
- Team and player statistics
- Game schedules and results

### User Experience
- Modern, responsive design
- Interactive game selection
- Confidence scoring system
- Detailed AI analysis display

## 🚀 Development

### Adding New Features
1. **Backend**: Add new endpoints in `api/main.py`
2. **Frontend**: Create components in `frontend/src/components/`
3. **API Integration**: Update `frontend/src/api/nflApi.ts`
4. **Types**: Update interfaces as needed

### Code Style
- **Python**: Follow PEP 8 standards
- **TypeScript**: Use strict mode and proper typing
- **React**: Functional components with hooks
- **CSS**: Tailwind CSS utility classes

## 📊 Performance

- **FastAPI**: High-performance Python web framework
- **Vercel**: Edge functions with global CDN
- **React 19**: Latest React with performance improvements
- **Optimized**: Lazy loading and efficient data fetching

## 🔒 Security

- CORS configuration for production
- Environment variable management
- Input validation with Pydantic
- Rate limiting (can be added)

## 🧪 Testing

### Backend Testing
```bash
cd api
pytest
```

### Frontend Testing
```bash
cd frontend
npm test
```

## 📈 Monitoring

- Vercel analytics and performance monitoring
- API health checks
- Error logging and monitoring
- Performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the Vercel deployment logs

## 🔄 Updates

### Recent Changes
- ✅ Complete backend rebuild with Python FastAPI
- ✅ nfl_data_py integration for rich NFL data
- ✅ Vercel deployment configuration
- ✅ Enhanced React frontend with TypeScript
- ✅ AI-powered prediction system
- ✅ Modern UI/UX with Tailwind CSS

---

**Gridiron Guru** - The future of NFL predictions is here. Powered by AI, backed by data. 🏈✨
# Force Vercel deployment
