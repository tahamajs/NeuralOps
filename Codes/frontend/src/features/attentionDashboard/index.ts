// src/features/attentionDashboard/index.ts
export * from './types/attention';
export * from './types';
export * from './stores/useAttentionDashboardStore';
export * from './useAttentionStore';

// Zone 1-4 Components
export * from './components/Zone1_ContextBar';
export * from './components/Zone2_AttentionCanvas';
export * from './components/Zone3_LiveMathPanel';
export * from './components/Zone4_ControlPanel';

// Zone 5-9 Components
export * from './components/HeadScatter3DPanel';
export * from './components/HeadAblationLab';
export * from './components/CircuitTracerPanel';
export * from './components/LogitLensPanel';
export * from './components/SemanticPathwayPanel';
export * from './components/GamificationSidebar';
export * from './components/SocialHubPanel';
export * from './components/CrossPlatformIntegrationPanel';

// Suites
export { AttentionCanvasSuite, default } from './AttentionCanvasSuite';
export { InterpretabilitySuite } from './InterpretabilitySuite';

