import React from 'react';
import { useNavigate } from 'react-router-dom';

interface HeaderProps {
  title?: string;
  subtitle?: string;
}

const Header: React.FC<HeaderProps> = ({ 
  title = "Gridiron Guru", 
  subtitle = "Home Page" 
}) => {
  const navigate = useNavigate();

  const handleSubtitleClick = () => {
    navigate('/');
  };

  return (
    <div className="relative w-full h-[453px] overflow-hidden">
      {/* Background Image with Gradient Overlay */}
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage: "url('/inside-header.png')"
        }}
      />
      
      {/* Gradient Overlay */}
      <div 
        className="absolute bottom-0 left-0 w-full h-[129px]"
        style={{
          background: 'linear-gradient(180deg, rgba(249, 250, 251, 0) 0%, #f9fafb 100%)'
        }}
      />
      
      {/* Main Title */}
      <div 
        className="absolute left-1/2 top-[80px] mobile:top-[100px] sm:top-[120px] lg:top-[159px] transform -translate-x-1/2 text-center font-rubik-dirt text-4.5xl mobile:text-7xl sm:text-8xl md:text-9xl lg:text-[100px] xl:text-[120px] leading-tight text-white w-[90vw] max-w-4xl"
        style={{
          textShadow: 'rgba(29, 41, 61, 0.1) 0px 20px 25px, rgba(29, 41, 61, 0.1) 0px 8px 10px, rgba(0, 0, 0, 0.25) 0px 4px 21.6px'
        }}
      >
        <div className="hidden mobile:block">
          {title}
        </div>
        <div className="block mobile:hidden">
          <div className="leading-none">
            Gridiron<br />Guru
          </div>
        </div>
      </div>
      
      {/* Subtitle */}
      <div 
        className="absolute left-1/2 top-[200px] mobile:top-[220px] sm:top-[250px] lg:top-[280px] transform -translate-x-1/2 text-center cursor-pointer hover:opacity-80 transition-opacity text-[#ffe08a] text-base mobile:text-lg sm:text-xl lg:text-2xl xl:text-[28px] font-semibold whitespace-nowrap"
        style={{
          fontFamily: 'Inter, sans-serif'
        }}
        onClick={handleSubtitleClick}
      >
        {subtitle}
      </div>
    </div>
  );
};

export default Header;
