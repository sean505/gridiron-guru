import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';

const HomeHeader: React.FC = () => {
  const navigate = useNavigate();

  const handlePredictionsClick = () => {
    navigate('/predictor');
  };

  return (
    <div className="relative w-full h-[890px] overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-[#d9d9d9]" />
      
      {/* Football Art Background Image */}
      <div 
        className="absolute top-[-77px] left-0 w-full h-[1088px] bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage: "url('/football-art-2.png')", // Place the image in frontend/public/football-art-2.png
          backgroundSize: "cover",
          width: "100%"
        }}
      />
      
      {/* Gradient Overlay */}
      <div 
        className="absolute inset-0 w-full h-full"
        style={{
          background: 'linear-gradient(270deg, rgba(217, 217, 217, 0) 29.937%, #d5d6da 82.317%)'
        }}
      />
      
      {/* Main Title */}
      <div 
        className="absolute left-4 mobile:left-8 sm:left-12 lg:left-16 top-[60px] mobile:top-[70px] sm:top-[80px] lg:top-[94px] text-white font-rubik-dirt text-4.5xl mobile:text-8xl sm:text-9xl md:text-[120px] lg:text-[140px] xl:text-[155px] leading-none"
        style={{
          textShadow: 'rgba(29, 41, 61, 0.1) 0px 20px 25px, rgba(29, 41, 61, 0.1) 0px 8px 10px, rgba(0, 0, 0, 0.25) 0px 4px 21.6px'
        }}
      >
        <div className="hidden mobile:block leading-none">
          Gridiron Guru
        </div>
        <div className="block mobile:hidden leading-none">
          Gridiron<br />Guru
        </div>
      </div>
      
      {/* Subtitle */}
      <div 
        className="absolute left-4 mobile:left-8 sm:left-12 lg:left-16 top-[280px] mobile:top-[320px] sm:top-[380px] lg:top-[453px] w-[90vw] max-w-2xl text-[#4a5565] text-lg mobile:text-xl sm:text-2xl md:text-3xl lg:text-4xl xl:text-[53px] leading-tight"
        style={{
          fontFamily: 'Heebo, sans-serif'
        }}
      >
        Weekly Game Predictions & Upset Picks Driven by AI
      </div>
      
      {/* CTA Button */}
      <button
        onClick={handlePredictionsClick}
        className="absolute left-4 mobile:left-8 sm:left-12 lg:left-16 top-[400px] mobile:top-[450px] sm:top-[550px] lg:top-[667px] bg-[#1447e6] text-white px-4 mobile:px-6 sm:px-8 py-2 mobile:py-3 sm:py-4 rounded-[10px] flex items-center justify-center gap-1 mobile:gap-2 shadow-[0px_5px_7.5px_rgba(29,41,61,0.1)] hover:bg-[#0f3bc7] transition-colors text-sm mobile:text-base sm:text-lg lg:text-xl font-bold"
        style={{
          fontFamily: 'Inter, sans-serif'
        }}
      >
        This Week's Predictions
        <ArrowRight className="w-4 h-4 mobile:w-5 mobile:h-5 sm:w-6 sm:h-6" />
      </button>
      
      {/* Bottom Gradient */}
      <div 
        className="absolute bottom-0 left-0 w-full h-[129px]"
        style={{
          background: 'linear-gradient(180deg, rgba(249, 250, 251, 0) 0%, #f9fafb 100%)'
        }}
      />
    </div>
  );
};

export default HomeHeader;
