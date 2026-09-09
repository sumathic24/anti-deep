import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
})

// ── Text Analysis ─────────────────────────────────────────────
export const analyzeText = async (text) => {
  const res = await api.post('/api/text/analyze', { text })
  return res.data
}

export const analyzeFakeNews = async (text) => {
  const res = await api.post('/api/text/fake-news', { text })
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
