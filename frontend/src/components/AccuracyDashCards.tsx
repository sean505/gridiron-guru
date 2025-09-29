import React from 'react';
import { Star, Target, TrendingUp, AlertTriangle } from 'lucide-react';

interface AccuracyStats {
  totalGames: number;
  correctPicks: number;
  accuracyPercentage: number;
  upsetPicksCorrect: number;
  upsetPicksTotal: number;
  upsetAccuracyPercentage: number;
  upsetThreatsCorrect: number;
  upsetThreatsTotal: number;
  upsetThreatsAccuracyPercentage: number;
}

interface AccuracyDashCardsProps {
  stats: AccuracyStats | null;
  weekInfo: {
    season: number;
    week: number;
    total_games: number;
  } | null;
}

const AccuracyDashCards: React.FC<AccuracyDashCardsProps> = ({ stats, weekInfo }) => {
  if (!stats || !weekInfo) {
    return null;
  }

  const formatPercentage = (percentage: number) => {
    return Math.round(percentage);
  };

  return (
    <div className="w-full max-w-6xl mx-auto px-4 mb-6">
      <div className="grid grid-cols-1 mobile:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Week Record Card */}
        <div className="bg-blue-600 rounded-xl p-4 text-white">
          <div className="flex items-center justify-between mb-3">
            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
              <Star className="w-4 h-4 text-blue-600" />
            </div>
            <h3 className="text-sm font-semibold text-blue-100">Week Record</h3>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold">{stats.correctPicks}</span>
            <span className="text-sm text-blue-200">of</span>
            <span className="text-2xl font-bold">{stats.totalGames}</span>
          </div>
          <div className="mt-2">
            <div className="bg-blue-500 rounded-full px-2 py-1 inline-block">
              <span className="text-sm font-semibold">{formatPercentage(stats.accuracyPercentage)}%</span>
            </div>
          </div>
        </div>

        {/* Upset Picks Card */}
        <div className="bg-yellow-600 rounded-xl p-4 text-white">
          <div className="flex items-center justify-between mb-3">
            <div className="w-8 h-8 bg-yellow-100 rounded-full flex items-center justify-center">
              <Target className="w-4 h-4 text-yellow-600" />
            </div>
            <h3 className="text-sm font-semibold text-yellow-100">Upset Picks</h3>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold">{stats.upsetPicksCorrect}</span>
            <span className="text-sm text-yellow-200">of</span>
            <span className="text-2xl font-bold">{stats.upsetPicksTotal}</span>
          </div>
          <div className="mt-2">
            <div className="bg-yellow-500 rounded-full px-2 py-1 inline-block">
              <span className="text-sm font-semibold">{formatPercentage(stats.upsetAccuracyPercentage)}%</span>
            </div>
          </div>
        </div>

        {/* Upset Threats Card */}
        <div className="bg-orange-600 rounded-xl p-4 text-white">
          <div className="flex items-center justify-between mb-3">
            <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center">
              <AlertTriangle className="w-4 h-4 text-orange-600" />
            </div>
            <h3 className="text-sm font-semibold text-orange-100">Upset Threats</h3>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold">{stats.upsetThreatsCorrect}</span>
            <span className="text-sm text-orange-200">of</span>
            <span className="text-2xl font-bold">{stats.upsetThreatsTotal}</span>
          </div>
          <div className="mt-2">
            <div className="bg-orange-500 rounded-full px-2 py-1 inline-block">
              <span className="text-sm font-semibold">{formatPercentage(stats.upsetThreatsAccuracyPercentage)}%</span>
            </div>
          </div>
        </div>

        {/* Overall Accuracy Card */}
        <div className="bg-green-600 rounded-xl p-4 text-white">
          <div className="flex items-center justify-between mb-3">
            <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
              <TrendingUp className="w-4 h-4 text-green-600" />
            </div>
            <h3 className="text-sm font-semibold text-green-100">Overall Accuracy</h3>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold">{formatPercentage(stats.accuracyPercentage)}%</span>
          </div>
          <div className="mt-2">
            <div className="bg-green-500 rounded-full px-2 py-1 inline-block">
              <span className="text-sm font-semibold">{stats.correctPicks} correct</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default AccuracyDashCards;
