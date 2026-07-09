// src/hooks/useApi.js
import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';

export function useApi(fetchFn, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    try {
      const result = await fetchFn();
      setData(result);
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, deps);

  useEffect(() => { refetch(); }, [refetch]);

  return { data, loading, error, refetch };
}

export function usePolling(fetchFn, intervalMs = 10000) {
  const { data, loading, error, refetch } = useApi(fetchFn);

  useEffect(() => {
    const timer = setInterval(refetch, intervalMs);
    return () => clearInterval(timer);
  }, [refetch, intervalMs]);

  return { data, loading, error, refetch };
}