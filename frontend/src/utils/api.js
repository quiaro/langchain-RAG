import axios from 'axios';

// Determine base URL based on environment
const isProduction = process.env.REACT_APP_PRODUCTION === 'true';
const baseURL = isProduction ? '' : 'http://0.0.0.0:7860';

// Create axios instance with base URL
const api = axios.create({
  baseURL,
});

// Export the API base URL for fetch requests
export const getApiUrl = (path) => {
  return `${baseURL}${path}`;
};

export default api;
