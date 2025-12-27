import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchSubjects = async () => {
  const response = await api.get('/api/subjects');
  return response.data;
};

export const fetchPredictions = async (subject, sampleIdx = null) => {
  const params = sampleIdx !== null ? { sample_idx: sampleIdx } : {};
  const response = await api.get(`/api/predictions/${subject}`, { params });
  return response.data;
};

export const fetchCortexMesh = async () => {
  const response = await api.get('/api/cortex-mesh');
  return response.data;
};

export const runPrediction = async (subject, checkpoint = 'checkpoints/best_model.pt', normalize = true) => {
  const response = await api.post(`/api/predict/${subject}`, null, {
    params: { checkpoint, normalize },
  });
  return response.data;
};

export const uploadAndPredict = async (file, checkpoint = 'checkpoints/best_model.pt', normalize = true) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/api/upload-and-predict', formData, {
    params: { checkpoint, normalize },
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get('/api/health');
  return response.data;
};

export default api;

