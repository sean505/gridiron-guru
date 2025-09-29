/**
 * Week management utilities for NFL season
 * Games shift every Tuesday at 11:59pm PT
 */

export interface WeekInfo {
  season: number;
  currentWeek: number;
  previousWeek: number;
  upcomingWeek: number;
  isTuesdayShift: boolean;
}

/**
 * Get the current NFL week based on Tuesday 11:59pm PT rule
 * Tuesday 11:59pm PT is when games shift between weeks
 */
export function getCurrentNFLWeek(): WeekInfo {
  const now = new Date();
  
  // Convert to Pacific Time
  const pacificTime = new Date(now.toLocaleString("en-US", {timeZone: "America/Los_Angeles"}));
  
  // Get current date components
  const year = pacificTime.getFullYear();
  const month = pacificTime.getMonth(); // 0-11
  const date = pacificTime.getDate();
  const dayOfWeek = pacificTime.getDay(); // 0 = Sunday, 1 = Monday, 2 = Tuesday, etc.
  const hours = pacificTime.getHours();
  const minutes = pacificTime.getMinutes();
  
  // NFL season typically starts in September (month 8)
  // For 2025 season, we'll assume it starts September 4th, 2025 (Week 1)
  const seasonStartDate = new Date(2025, 8, 4); // September 4, 2025
  const seasonStartWeek = 1;
  
  // Calculate days since season start
  const daysSinceStart = Math.floor((pacificTime.getTime() - seasonStartDate.getTime()) / (1000 * 60 * 60 * 24));
  
  // Calculate current week (each week starts on Tuesday 11:59pm PT)
  // Week 1: Sep 4-10, 2025
  // Week 2: Sep 11-17, 2025 (starts Tuesday Sep 10 at 11:59pm PT)
  // etc.
  
  let currentWeek = Math.floor(daysSinceStart / 7) + seasonStartWeek;
  
  // Check if we're past Tuesday 11:59pm PT this week
  // If it's Wednesday or later, or if it's Tuesday after 11:59pm, we're in the next week
  const isTuesdayShift = (dayOfWeek === 2 && hours >= 23 && minutes >= 59) || dayOfWeek > 2;
  
  if (isTuesdayShift) {
    currentWeek += 1;
  }
  
  // No shift needed - the calculation is correct as is
  // Previous Week will show N-1, Current Week will show N, Upcoming Week will show N+1
  
  // Ensure we don't go beyond week 18 (regular season) or week 22 (including playoffs)
  currentWeek = Math.min(currentWeek, 22);
  
  // Calculate previous and upcoming weeks
  const previousWeek = Math.max(currentWeek - 1, 1);
  const upcomingWeek = Math.min(currentWeek + 1, 22);
  
  return {
    season: year,
    currentWeek,
    previousWeek,
    upcomingWeek,
    isTuesdayShift
  };
}

/**
 * Get week display name
 */
export function getWeekDisplayName(week: number): string {
  if (week <= 18) {
    return `Week ${week}`;
  } else if (week === 19) {
    return 'Wild Card';
  } else if (week === 20) {
    return 'Divisional';
  } else if (week === 21) {
    return 'Conference';
  } else if (week === 22) {
    return 'Super Bowl';
  }
  return `Week ${week}`;
}

/**
 * Check if a week is in the past, current, or future
 */
export function getWeekStatus(week: number): 'past' | 'current' | 'future' {
  const weekInfo = getCurrentNFLWeek();
  
  if (week < weekInfo.currentWeek) {
    return 'past';
  } else if (week === weekInfo.currentWeek) {
    return 'current';
  } else {
    return 'future';
  }
}

/**
 * Get the appropriate week for each section
 */
export function getWeekForSection(section: 'previous' | 'current' | 'upcoming'): number {
  const weekInfo = getCurrentNFLWeek();
  
  switch (section) {
    case 'previous':
      return weekInfo.previousWeek;
    case 'current':
      return weekInfo.currentWeek;
    case 'upcoming':
      return weekInfo.upcomingWeek;
    default:
      return weekInfo.currentWeek;
  }
}
