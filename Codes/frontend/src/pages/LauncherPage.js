import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import api from '../api/client';
import ModelPicker from '../components/ModelPicker';
import TaskPicker from '../components/TaskPicker';
import AttentionPicker from '../components/AttentionPicker';
import MethodPicker from '../components/MethodPicker';
import HyperparameterFields from '../components/HyperparameterFields';
import PresetSelector from '../components/PresetSelector';
import ConfigPreviewPanel from '../components/ConfigPreviewPanel';
import LaunchButton from '../components/LaunchButton';
export default function LauncherPage() {
    const [tab, setTab] = useState('training');
    const [form, setForm] = useState({
        name: '',
        model_name: 'distilbert-base-uncased',
        task_name: 'mrpc',
        attention_type: 'full',
        method: 'baseline',
        compression_rank: 64,
        epochs: 3,
        batch_size: 16,
        learning_rate: 2e-5,
        max_seq_length: 128,
        fp16: false,
        seed: 42,
        target_rank: 64,
        compression_method: 'tt',
    });
    const navigate = useNavigate();
    const mutation = useMutation({
        mutationFn: async () => {
            const endpoint = tab === 'training' ? '/train/' : tab === 'evaluation' ? '/evaluate/' : '/compress/';
            const res = await api.post(endpoint, form);
            return res.data;
        },
        onSuccess: (data) => {
            navigate(`/experiments/${data.id}`);
        },
    });
    const updateField = (key, value) => setForm((prev) => ({ ...prev, [key]: value }));
    return (_jsxs("div", { className: "space-y-8 max-w-5xl mx-auto", children: [_jsxs("div", { className: "flex flex-col gap-2", children: [_jsx("h1", { className: "text-3xl font-extrabold text-slate-900 tracking-tight", children: "Experiment Launcher" }), _jsx("p", { className: "text-slate-500", children: "Configure and deploy new attention compression jobs to the cluster." })] }), _jsx("div", { className: "flex p-1 bg-slate-200/50 backdrop-blur rounded-xl w-fit", children: ['training', 'evaluation', 'compression'].map((t) => (_jsx("button", { onClick: () => setTab(t), className: `px-6 py-2 rounded-lg text-sm font-semibold transition-all ${tab === t
                        ? 'bg-white text-blue-600 shadow-sm'
                        : 'text-slate-600 hover:text-slate-900'}`, children: t.charAt(0).toUpperCase() + t.slice(1) }, t))) }), _jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-3 gap-8", children: [_jsx("div", { className: "lg:col-span-2 space-y-6", children: _jsxs("div", { className: "card space-y-6", children: [_jsxs("div", { className: "flex items-center gap-2 pb-4 border-b border-slate-100", children: [_jsx("div", { className: "w-2 h-2 rounded-full bg-blue-600" }), _jsx("h2", { className: "font-bold text-slate-800 uppercase tracking-wider text-xs", children: "Arithemtic Configuration" })] }), _jsx("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-6", children: tab === 'training' && (_jsxs(_Fragment, { children: [_jsx(ModelPicker, { value: form.model_name, onChange: (v) => updateField('model_name', v) }), _jsx(TaskPicker, { value: form.task_name, onChange: (v) => updateField('task_name', v) }), _jsx("div", { className: "md:col-span-2", children: _jsx(AttentionPicker, { value: form.attention_type, onChange: (v) => updateField('attention_type', v) }) }), _jsx(MethodPicker, { value: form.method, onChange: (v) => updateField('method', v) }), _jsx(HyperparameterFields, { values: form, onChange: updateField })] })) }), _jsx("div", { className: "pt-6", children: _jsx(LaunchButton, { onClick: () => mutation.mutate(), loading: mutation.isPending, disabled: !form.name }) })] }) }), _jsxs("div", { className: "space-y-6", children: [_jsx(ConfigPreviewPanel, { config: form }), _jsx(PresetSelector, { onApply: (preset) => setForm({ ...form, ...preset }) })] })] })] }));
}
className = "w-full border rounded px-3 py-2";
value = { form, : .compression_method };
onChange = {}(e);
updateField('compression_method', e.target.value);
    >
        (_jsx("option", { value: "tt", children: "Tensor Train (tt)" })
            ,
                _jsx("option", { value: "svd", children: "SVD" })
                    ,
                        _jsx("option", { value: "soka", children: "SoKA" }));
select >
;
div >
    _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium", children: "Target Rank" }), _jsx("input", { type: "number", className: "w-full border rounded px-3 py-2", value: form.target_rank, onChange: (e) => updateField('target_rank', parseInt(e.target.value)) })] });
 >
;
div >
    (_jsx(ConfigPreviewPanel, { config: form })
        ,
            _jsx(LaunchButton, { loading: mutation.isPending, onClick: () => mutation.mutate() }));
div >
;
