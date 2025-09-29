/**
 * Utility functions for upset categorization and display
 */

export type UpsetLevel = 'none' | 'threat' | 'pick';

export interface UpsetInfo {
  level: UpsetLevel;
  label: string;
  message: string;
  colorClass: string;
  bgColorClass: string;
  borderColorClass: string;
}

/**
 * Categorize upset potential into threat levels
 */
export function categorizeUpset(upsetPotential: number): UpsetInfo {
  if (upsetPotential >= 40) {
    return {
      level: 'pick',
      label: 'UPSET PICK',
      message: 'Strong upset candidate - model favors underdog scenario',
      colorClass: 'text-red-700',
      bgColorClass: 'bg-red-50',
      borderColorClass: 'border-red-200'
    };
  } else if (upsetPotential >= 25) {
    return {
      level: 'threat',
      label: 'UPSET THREAT',
      message: 'This game has upset potential - watch for surprises',
      colorClass: 'text-orange-600',
      bgColorClass: 'bg-orange-50',
      borderColorClass: 'border-orange-200'
    };
  } else {
    return {
      level: 'none',
      label: '',
      message: '',
      colorClass: 'text-gray-600',
      bgColorClass: 'bg-gray-50',
      borderColorClass: 'border-gray-200'
    };
  }
}

/**
 * Get upset potential color for badges
 */
export function getUpsetPotentialColor(upsetPotential: number): string {
  if (upsetPotential >= 40) {
    return 'bg-red-100 text-red-800 border-red-200';
  } else {
    return 'bg-yellow-100 text-amber-800 border-yellow-300';
  }
}

/**
 * Get team highlight styling for upset scenarios
 */
export function getTeamUpsetStyling(upsetLevel: UpsetLevel, isPredictedWinner: boolean): string {
  if (upsetLevel === 'pick' && !isPredictedWinner) {
    // High upset potential - highlight the underdog
    return 'bg-red-50 border border-red-200 rounded-lg';
  } else if (upsetLevel === 'threat' && !isPredictedWinner) {
    // Medium upset potential - subtle highlight
    return 'bg-orange-50 border border-orange-200 rounded-lg';
  }
  return '';
}

/**
 * Get prediction indicator styling
 */
export function getPredictionIndicatorClass(upsetLevel: UpsetLevel): string {
  if (upsetLevel === 'pick') {
    return 'bg-red-500';
  } else if (upsetLevel === 'threat') {
    return 'bg-orange-500';
  }
  return 'bg-blue-500';
}

/**
 * Get team status text for upset scenarios
 */
export function getTeamStatusText(upsetLevel: UpsetLevel, isPredictedWinner: boolean): string {
  if (isPredictedWinner) {
    return 'Predict to Win';
  } else if (upsetLevel === 'pick') {
    return 'Upset Pick!';
  } else if (upsetLevel === 'threat') {
    return 'Upset Threat';
  }
  return '';
}

/**
 * Get team status text color class
 */
export function getTeamStatusColorClass(upsetLevel: UpsetLevel, isPredictedWinner: boolean): string {
  if (isPredictedWinner) {
    return 'text-blue-600';
  } else if (upsetLevel === 'pick') {
    return 'text-red-600';
  } else if (upsetLevel === 'threat') {
    return 'text-yellow-600';
  }
  return 'text-gray-600';
}
