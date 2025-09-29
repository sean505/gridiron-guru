/**
 * Ultra-simple API URL detection
 * This is a completely fresh approach to avoid any caching issues
 */

export const getApiUrl = (): string => {
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  
  // Check environment variable but ignore localhost values when we're on external IP
  const envUrl = (import.meta as any).env?.VITE_API_URL;
  
  // If we're on external IP and env is localhost, ignore the env variable
  if (envUrl && envUrl.includes('localhost') && hostname !== 'localhost' && hostname !== '127.0.0.1') {
    // Ignore localhost env variable for external IP access
  } else if (envUrl && !envUrl.includes('localhost')) {
    return envUrl;
  }
  
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  } else {
    return `${protocol}//${hostname}:8000`;
  }
};

export const API_URL = getApiUrl();
