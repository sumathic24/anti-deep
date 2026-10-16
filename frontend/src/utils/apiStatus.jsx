import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { checkHealth } from './api'

const POLL_MS = 30000
const ApiStatusContext = createContext({ status: 'checking', recheck: () => {} })

/** Polls the backend's /health so every page knows whether the API is reachable. */
export function ApiStatusProvider({ children }) {
  const [status, setStatus] = useState('checking') // checking | online | offline

  const recheck = useCallback(() => {
    return checkHealth()
      .then((ok) => setStatus(ok ? 'online' : 'offline'))
      .catch(() => setStatus('offline'))
  }, [])

  useEffect(() => {
    recheck()
    const timer = setInterval(recheck, POLL_MS)
    return () => clearInterval(timer)
  }, [recheck])

  return <ApiStatusContext.Provider value={{ status, recheck }}>{children}</ApiStatusContext.Provider>
}

export const useApiStatus = () => useContext(ApiStatusContext)
