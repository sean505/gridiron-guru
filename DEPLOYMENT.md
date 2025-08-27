# 🚀 Gridiron Guru Deployment Guide

## Overview
This guide will walk you through deploying your Gridiron Guru application to Vercel, where both the React frontend and Python FastAPI backend will run seamlessly.

## 🎯 What We're Deploying

- **Frontend**: React 19 + TypeScript + Tailwind CSS
- **Backend**: Python FastAPI with nfl_data_py integration
- **Platform**: Vercel (Frontend + API Functions)

## 📋 Prerequisites

1. **GitHub Account**: Your code should be in a GitHub repository
2. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
3. **OpenAI API Key**: For AI predictions
4. **Node.js & npm**: For local development

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Repository

Ensure your code is committed and pushed to GitHub:
```bash
git add .
git commit -m "Ready for Vercel deployment"
git push origin main
```

### Step 2: Deploy to Vercel

#### Option A: Deploy via Vercel Dashboard (Recommended)

1. **Go to [vercel.com](https://vercel.com) and sign in**
2. **Click "New Project"**
3. **Import your GitHub repository**
4. **Configure the project:**
   - **Framework Preset**: Other
   - **Root Directory**: `./` (root of your project)
   - **Build Command**: `cd frontend && npm run build`
   - **Output Directory**: `frontend/dist`
   - **Install Command**: `cd frontend && npm install`

#### Option B: Deploy via Vercel CLI

```bash
# Install Vercel CLI globally (if you have permissions)
npm install -g vercel

# Or use the local version
cd frontend
npx vercel

# Follow the prompts to:
# 1. Link to your Vercel account
# 2. Set up the project
# 3. Configure settings
```

### Step 3: Configure Environment Variables

In your Vercel dashboard, go to **Settings > Environment Variables** and add:

```env
# Required for AI predictions
OPENAI_API_KEY=your_openai_api_key_here

# NFL data configuration
CURRENT_SEASON=2024

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### Step 4: Configure Build Settings

In your Vercel project settings:

1. **Build & Development Settings:**
   - **Framework Preset**: Other
   - **Build Command**: `cd frontend && npm run build`
   - **Output Directory**: `frontend/dist`
   - **Install Command**: `cd frontend && npm install`

2. **Functions:**
   - **Python Version**: 3.9
   - **Node.js Version**: 18.x

### Step 5: Deploy

1. **Click "Deploy"**
2. **Wait for build to complete**
3. **Your app will be live at your Vercel URL**

## 🔧 Post-Deployment Configuration

### 1. Test Your API Endpoints

Your API will be available at:
- `https://your-project.vercel.app/api/`
- `https://your-project.vercel.app/api/teams`
- `https://your-project.vercel.app/api/games`
- `https://your-project.vercel.app/api/predict`

### 2. Update Frontend API URL

Once deployed, update your frontend environment:
```env
VITE_API_URL=https://your-project.vercel.app
```

### 3. Test the Full Application

1. **Visit your Vercel URL**
2. **Navigate to `/predictor`**
3. **Test the prediction functionality**
4. **Verify API responses**

## 🐛 Troubleshooting

### Common Issues

#### 1. Build Failures
- **Error**: "Module not found"
- **Solution**: Ensure all dependencies are in `frontend/package.json`

#### 2. API Not Working
- **Error**: "Function not found"
- **Solution**: Check `vercel.json` configuration and Python runtime

#### 3. Environment Variables
- **Error**: "API key not found"
- **Solution**: Verify environment variables in Vercel dashboard

#### 4. Python Dependencies
- **Error**: "Import error"
- **Solution**: Check `api/requirements.txt` and Python version

### Debug Steps

1. **Check Vercel build logs**
2. **Verify environment variables**
3. **Test API endpoints individually**
4. **Check Python function logs**

## 📊 Monitoring & Analytics

### Vercel Dashboard Features
- **Real-time Analytics**: Monitor traffic and performance
- **Function Logs**: Debug API issues
- **Performance Metrics**: Track response times
- **Error Tracking**: Monitor application errors

### Health Checks
- **API Health**: `/api/health`
- **Frontend**: Your main Vercel URL
- **Functions**: Individual API endpoints

## 🔄 Continuous Deployment

### Automatic Deployments
- **GitHub Integration**: Automatic deployments on push
- **Preview Deployments**: Test changes before production
- **Rollback**: Easy rollback to previous versions

### Development Workflow
1. **Make changes locally**
2. **Test with mock API**
3. **Push to GitHub**
4. **Vercel auto-deploys**
5. **Test production deployment**

## 🎉 Success Checklist

- [ ] Frontend loads without errors
- [ ] API endpoints respond correctly
- [ ] AI predictions work
- [ ] Environment variables are set
- [ ] Build completes successfully
- [ ] All functions are accessible

## 🆘 Getting Help

### Resources
- **Vercel Documentation**: [vercel.com/docs](https://vercel.com/docs)
- **FastAPI Documentation**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **React Documentation**: [react.dev](https://react.dev)

### Support
- **Vercel Support**: Available in dashboard
- **GitHub Issues**: For code-related problems
- **Community**: Stack Overflow, Discord, etc.

## 🚀 Next Steps After Deployment

1. **Set up custom domain** (optional)
2. **Configure monitoring and alerts**
3. **Set up CI/CD pipeline**
4. **Add analytics and tracking**
5. **Implement caching strategies**
6. **Add rate limiting and security**

---

**🎯 Your Gridiron Guru app is now ready for the world!** 

Deploy with confidence and start making AI-powered NFL predictions! 🏈✨
