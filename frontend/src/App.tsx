import React from 'react';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./pages/Home";
import WeeklyPredictor from "./components/WeeklyPredictor";
import Simulation from "./pages/Simulation";
import PreviousWeek from "./pages/PreviousWeek";

export default function App() {
  console.log('🏈 App component is mounting!');
  
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/predictor" element={<WeeklyPredictor />} />
        <Route path="/simulation" element={<Simulation />} />
        <Route path="/previous-week" element={<PreviousWeek />} />
      </Routes>
    </Router>
  );
}
