import React from 'react';

const PredictionCardSkeleton: React.FC = () => {
  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden animate-pulse">
      {/* Banner */}
      <div className="h-12 bg-gray-200 rounded-t-xl"></div>
      
      {/* Main Content */}
      <div className="p-6">
        {/* Teams Section */}
        <div className="flex justify-center gap-4 lg:gap-6 mb-6 lg:mb-8">
          {/* Away Team */}
          <div className="flex flex-col items-center gap-1 w-40 lg:w-48 px-0 py-2">
            <div className="h-4 bg-gray-200 rounded w-20 mb-2"></div>
            <div className="h-6 bg-gray-200 rounded w-24 mb-2"></div>
            <div className="h-4 bg-gray-200 rounded w-8"></div>
          </div>
          
          {/* VS */}
          <div className="flex flex-col items-center justify-center">
            <div className="h-6 bg-gray-200 rounded w-8 mb-2"></div>
            <div className="h-4 bg-gray-200 rounded w-12"></div>
          </div>
          
          {/* Home Team */}
          <div className="flex flex-col items-center gap-1 w-40 lg:w-48 px-0 py-2">
            <div className="h-4 bg-gray-200 rounded w-20 mb-2"></div>
            <div className="h-6 bg-gray-200 rounded w-24 mb-2"></div>
            <div className="h-4 bg-gray-200 rounded w-8"></div>
          </div>
        </div>
        
        {/* Game Info */}
        <div className="flex flex-col items-center gap-1 w-40 lg:w-48 mx-auto mb-6">
          <div className="h-5 bg-gray-200 rounded w-20 mb-2"></div>
          <div className="h-4 bg-gray-200 rounded w-16 mb-2"></div>
          <div className="h-8 bg-gray-200 rounded w-32"></div>
        </div>
        
        {/* AI Analysis Section */}
        <div className="border border-t-0 rounded-b-xl p-4 shadow-sm bg-gray-50 border-gray-200">
          <div className="flex flex-col gap-2 mb-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-gray-200 rounded-full"></div>
              <div className="h-4 bg-gray-200 rounded w-48"></div>
            </div>
            <div className="space-y-2">
              <div className="h-3 bg-gray-200 rounded w-full"></div>
              <div className="h-3 bg-gray-200 rounded w-3/4"></div>
              <div className="h-3 bg-gray-200 rounded w-1/2"></div>
            </div>
          </div>
          
          <div className="flex justify-between items-center">
            <div className="bg-gray-200 border border-gray-300 rounded-xl px-2 py-1 flex items-center gap-2">
              <div className="h-4 bg-gray-300 rounded w-8"></div>
              <div className="w-px h-3 bg-gray-300"></div>
              <div className="h-4 bg-gray-300 rounded w-16"></div>
            </div>
            
            <div className="bg-gray-200 border border-gray-300 rounded-xl px-2 py-1 flex items-center gap-2">
              <div className="h-4 bg-gray-300 rounded w-8"></div>
              <div className="w-px h-3 bg-gray-300"></div>
              <div className="h-4 bg-gray-300 rounded w-20"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionCardSkeleton;