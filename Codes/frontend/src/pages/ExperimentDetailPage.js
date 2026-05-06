import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { useWebSocket } from '../hooks/useWebSocket';
import { useLiveStore } from '../store/experimentStore';
import api from '../api/client';
import StatusBadge from '../components/StatusBadge';
import ProgressBar from '../components/ProgressBar';
import LiveLossPlot from '../components/LiveLossPlot';
import LiveLogViewer from '../components/LiveLogViewer';
import ConfigSummary from '../components/ConfigSummary';
import ArtifactList from '../components/ArtifactList';
import DownloadModelButton from '../components/DownloadModelButton';
export default function ExperimentDetailPage() {
    const { id } = useParams();
    useWebSocket(id);
    const { data: experiment } = useQuery({
        queryKey: ['experiment', id],
        queryFn: async () => (await api.get(`/experiments/${id}/`)).data,
        enabled: !!id,
    });
    const live = useLiveStore();
    if (!experiment)
        return _jsx("div", { children: "Loading..." });
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { className: "flex justify-between items-center", children: [_jsx("h1", { className: "text-2xl font-bold", children: experiment.name || 'Experiment' }), _jsx(StatusBadge, { status: live.status || experiment.status })] }), _jsx(ProgressBar, { progress: live.progress }), _jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-2 gap-6", children: [_jsx(LiveLossPlot, { metrics: live.metrics }), _jsx(ConfigSummary, { config: experiment.config })] }), _jsx(LiveLogViewer, { logs: live.logs }), _jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-2 gap-6", children: [_jsx(ArtifactList, { experimentId: id }), _jsx(DownloadModelButton, { experimentId: id })] })] }));
}
