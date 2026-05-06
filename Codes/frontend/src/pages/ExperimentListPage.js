import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import api from '../api/client';
import ExperimentFilters from '../components/ExperimentFilters';
import ExperimentTable from '../components/ExperimentTable';
import PaginationControls from '../components/PaginationControls';
export default function ExperimentListPage() {
    const [filter, setFilter] = useState({ status: '', type: '', search: '' });
    const [page, setPage] = useState(1);
    const pageSize = 15;
    const { data: experiments = [] } = useQuery({
        queryKey: ['experiments'],
        queryFn: async () => (await api.get('/experiments/')).data,
        refetchInterval: 5000,
    });
    const filtered = experiments.filter((e) => {
        if (filter.status && e.status !== filter.status)
            return false;
        if (filter.type && e.job_type !== filter.type)
            return false;
        if (filter.search && !e.name?.toLowerCase().includes(filter.search.toLowerCase()))
            return false;
        return true;
    });
    const totalPages = Math.ceil(filtered.length / pageSize);
    const paged = filtered.slice((page - 1) * pageSize, page * pageSize);
    return (_jsxs("div", { className: "space-y-4", children: [_jsx("h1", { className: "text-2xl font-bold text-gray-800", children: "Experiments" }), _jsx(ExperimentFilters, { value: filter, onChange: setFilter }), _jsx(ExperimentTable, { experiments: paged }), _jsx(PaginationControls, { page: page, total: totalPages, onChange: setPage })] }));
}
