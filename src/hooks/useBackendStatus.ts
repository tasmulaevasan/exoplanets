import { useState, useEffect } from 'react';
import { api } from '@/lib/api';

export type BackendStatus = 'connected' | 'disconnected' | 'checking';

export function useBackendStatus(intervalMs: number = 30000) {
  const [status, setStatus] = useState<BackendStatus>('checking');
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  useEffect(() => {
    let isMounted = true;

    const checkStatus = async () => {
      try {
        await api.healthCheck();
        if (isMounted) {
          setStatus('connected');
          setLastChecked(new Date());
        }
      } catch (error) {
        if (isMounted) {
          setStatus('disconnected');
          setLastChecked(new Date());
        }
      }
    };

    // Initial check
    checkStatus();

    // Set up interval for periodic checks
    const interval = setInterval(checkStatus, intervalMs);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [intervalMs]);

  return { status, lastChecked };
}
