import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120000, // SHAP explanations on transformer models can take a while on CPU
})

// Human-readable message for a failed request
export const getErrorMessage = (err) => {
  if (err.response) {
    const detail = err.response.data?.detail
    if (typeof detail === 'string') return detail
    return `Server error (${err.response.status}). Please try again.`
  }
  if (err.code === 'ECONNABORTED') return 'The server took too long to respond. Please try again.'
  return `Cannot reach the backend at ${BASE_URL}. Make sure the backend server is running.`
}

// ── Health ────────────────────────────────────────────────────
export const checkHealth = async () => {
  const res = await api.get('/health', { timeout: 5000 })
  return res.data?.status === 'ok'
}

// ── Text Analysis ─────────────────────────────────────────────
export const analyzeText = async (text) => {
  const res = await api.post('/api/text/analyze', { text })
  return res.data
}

// ── Image Analysis ────────────────────────────────────────────
export const analyzeImage = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  const res = await api.post('/api/image/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

// ── Dashboard Stats ───────────────────────────────────────────
export const getDashboardStats = async () => {
  const res = await api.get('/api/stats')
  return res.data
}

export default api
