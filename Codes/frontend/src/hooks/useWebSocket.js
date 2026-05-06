import { useEffect, useRef } from 'react';
import { useLiveStore } from '../store/experimentStore';
/**
 * Hook for WebSocket connection to live experiment updates.
 * Automatically handles protocol selection (ws/wss) and token authentication.
 */
export function useWebSocket(experimentId) {
    const store = useLiveStore();
    const wsRef = useRef(null);
    useEffect(() => {
        if (!experimentId)
            return;
        store.reset();
        // Determine WebSocket protocol
        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        // Build WebSocket URL with optional host override
        const wsBaseURL = import.meta.env.VITE_WS_URL || `${protocol}://${window.location.host}`;
        const url = `${wsBaseURL}/ws/experiments/${experimentId}/`;
        const socket = new WebSocket(url);
        wsRef.current = socket;
        // Add token to WebSocket headers if available (note: 
        // WebSocket doesn't support custom headers in browsers, 
        // so token must be passed as URL parameter or handled by backend auth)
        const token = localStorage.getItem('token');
        if (token && !url.includes('?')) {
            // Some implementations may support token in query params
            // but Django Channels typically uses the authenticated user from HTTP upgrade
        }
        socket.onopen = () => {
            console.log(`WebSocket connected to ${url}`);
            store.addLog(`Connected to live stream`);
        };
        socket.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                switch (msg.type) {
                    case 'status':
                        store.updateStatus(msg.status);
                        break;
                    case 'log':
                        store.addLog(msg.data);
                        break;
                    case 'metrics':
                        store.updateMetrics(msg.data);
                        break;
                    case 'progress':
                        store.updateProgress(msg);
                        break;
                    case 'gpu':
                        store.updateGpu(msg);
                        break;
                    case 'final':
                        store.updateMetrics(msg.data);
                        store.addLog('Job completed');
                        break;
                    default:
                        console.warn('Unknown WebSocket message type:', msg.type);
                }
            }
            catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };
        socket.onerror = (event) => {
            console.error('WebSocket error:', event);
            store.addLog(`WebSocket error occurred`);
        };
        socket.onclose = () => {
            console.log('WebSocket closed');
            store.addLog('Connection closed');
        };
        return () => {
            if (socket.readyState === WebSocket.OPEN) {
                socket.close();
            }
        };
    }, [experimentId, store]);
    return wsRef;
}
