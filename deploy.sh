#!/bin/bash

echo "🚀 Gridiron Guru Deployment Script"
echo "=================================="

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
else
    echo "✅ Vercel CLI found"
fi

# Check if we're in the right directory
if [ ! -f "vercel.json" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

echo ""
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "🐍 Installing backend dependencies..."
cd api
pip install -r requirements.txt
cd ..

echo ""
echo "🔧 Setting up environment variables..."
if [ ! -f "api/.env" ]; then
    echo "⚠️  Please create api/.env file with your OpenAI API key"
    echo "   Copy from api/env.example and add your OPENAI_API_KEY"
fi

echo ""
echo "🚀 Deploying to Vercel..."
vercel --prod

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your app should now be live on Vercel"
echo ""
echo "📚 Next steps:"
echo "   1. Set environment variables in Vercel dashboard"
echo "   2. Test your API endpoints"
echo "   3. Customize your frontend as needed"
