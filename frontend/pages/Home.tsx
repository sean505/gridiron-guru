import WeeklyPredictor from '@/react-app/components/WeeklyPredictor';
import { Zap, Brain, Trophy } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 to-blue-900/20 opacity-20"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 py-16 sm:py-24">
          <div className="text-center mb-16">
            <div className="flex items-center justify-center gap-3 mb-6">
              <div className="bg-gradient-to-r from-green-400 to-blue-500 p-3 rounded-full shadow-lg">
                <span className="text-2xl">🏈</span>
              </div>
              <h1 className="text-5xl sm:text-6xl font-bold bg-gradient-to-r from-white via-purple-200 to-blue-200 bg-clip-text text-transparent">
                Gridiron Guru
              </h1>
            </div>
            
            <p className="text-xl sm:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto leading-relaxed">
              Your weekly football AI companion for predictions, creativity, and commentary all in one place
            </p>

            {/* Feature highlights */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-4xl mx-auto mb-12">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                <Brain className="w-10 h-10 text-purple-400 mx-auto mb-3" />
                <h3 className="text-lg font-semibold text-white mb-2">AI Predictions</h3>
                <p className="text-gray-300 text-sm">Get expert AI analysis and win probability for every game</p>
              </div>

              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                <Zap className="w-10 h-10 text-yellow-400 mx-auto mb-3" />
                <h3 className="text-lg font-semibold text-white mb-2">Upset Alerts</h3>
                <p className="text-gray-300 text-sm">Spot potential upsets and make bold predictions</p>
              </div>

              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                <Trophy className="w-10 h-10 text-green-400 mx-auto mb-3" />
                <h3 className="text-lg font-semibold text-white mb-2">Track Progress</h3>
                <p className="text-gray-300 text-sm">Compete against AI and track your prediction accuracy</p>
              </div>
            </div>

            <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-2xl p-1 inline-block shadow-2xl">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl px-6 py-3 border border-white/20">
                <span className="text-white font-semibold">🏈 Week&apos;s games updated every Tuesday</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="relative bg-white">
        <div className="absolute inset-0 bg-gradient-to-t from-white via-gray-50 to-white"></div>
        <div className="relative">
          <WeeklyPredictor />
        </div>
      </div>

      {/* Footer */}
      <div className="bg-slate-900 text-white py-12">
        <div className="max-w-6xl mx-auto px-4 text-center">
          <div className="flex items-center justify-center gap-2 mb-4">
            <span className="text-xl">🏈</span>
            <span className="text-xl font-bold">Gridiron Guru</span>
          </div>
          <p className="text-gray-400">
            Made with ⚡ for football fans who love the game and the numbers behind it
          </p>
        </div>
      </div>
    </div>
  );
}
