import { useEffect, useState } from 'react'

// Kept in memory for the browser tab, so a page's input and results survive switching
// to another page and back (a full reload still starts fresh).
const store = new Map()

export function usePersistentState(key, initialValue) {
  const [value, setValue] = useState(() => (store.has(key) ? store.get(key) : initialValue))
  useEffect(() => { store.set(key, value) }, [key, value])
  return [value, setValue]
}
