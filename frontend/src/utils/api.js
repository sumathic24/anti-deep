import axios from 'axios'

export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 180000, // SHAP explanations on large models can take a while on CPU
})

export const isCancel = (err) => axios.isCancel(err) || err?.code === 'ERR_CANCELED'

// Human-readable message for a failed request
export const getErrorMessage = (err) => {
  if (err.response) {
    const detail = err.response.data?.detail
    if (typeof detail === 'string') return detail
    return `Server error (${err.response.status}). Please try again.`
  }
  if (err.code === 'ECONNABORTED') return 'The server took too long to respond. Please try again.'
  return `Cannot reach the backend at ${API_BASE}. Make sure the backend server is running.`
}

// ── Health ────────────────────────────────────────────────────
export const checkHealth = async () => {
  const res = await api.get('/health', { timeout: 5000 })
  return res.data?.status === 'ok'
}

// ── Text Analysis ─────────────────────────────────────────────
export const analyzeText = async (text, { explain = true, signal } = {}) => {
  const res = await api.post('/api/text/analyze', { text, explain }, { signal })
  return res.data
}

// ── Image Analysis ────────────────────────────────────────────
export const analyzeImage = async (file, { signal, onUploadProgress } = {}) => {
  const formData = new FormData()
  formData.append('file', file)
  const res = await api.post('/api/image/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    signal,
    onUploadProgress: onUploadProgress && ((e) => e.total && onUploadProgress(e.loaded / e.total)),
  })
  return res.data
}

// ── Dashboard ─────────────────────────────────────────────────
export const getDashboardStats = async (days = 7) => {
  const res = await api.get('/api/stats', { params: { days } })
  return res.data
}

export const getHistory = async (limit = 10, offset = 0) => {
  const res = await api.get('/api/history', { params: { limit, offset } })
  return res.data
}

export const clearHistory = async () => {
  const res = await api.delete('/api/history')
  return res.data
}

export default api
