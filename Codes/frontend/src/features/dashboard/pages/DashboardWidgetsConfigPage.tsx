import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'

export default function DashboardWidgetsConfigPage() {
  const queryClient = useQueryClient()
  const { data: widgets = [], isLoading } = useQuery({
    queryKey: ['dashboardWidgets'],
    queryFn: async () => (await api.get('/api/widgets/')).data,
  })

  const saveMutation = useMutation({
    mutationFn: async (layout: any) => api.put('/api/widgets/layout/', { widgets: layout }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['dashboardWidgets'] }),
  })

  const handleToggle = (id: string) => {
    const updated = widgets.map((w: any) =>
      w.id === id ? { ...w, is_visible: !w.is_visible } : w
    )
    saveMutation.mutate(updated)
  }

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        DASHBOARD <span className="text-neon-blue">WIDGETS</span>
      </h1>
      <Card>
        <p className="text-slate-400 text-xs mb-4">
          Toggle visibility of dashboard widgets. Drag‑and‑drop coming soon.
        </p>
        <div className="grid gap-4">
          {widgets.map((w: any) => (
            <div key={w.id} className="flex items-center justify-between p-3 bg-cyber-800 rounded-lg">
              <div>
                <span className="text-white font-semibold text-sm">{w.title || w.widget_type}</span>
                <p className="text-[10px] text-slate-500">{w.widget_type}</p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" className="sr-only peer" checked={w.is_visible} onChange={() => handleToggle(w.id)} />
                <div className="w-9 h-5 bg-cyber-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-neon-blue"></div>
              </label>
            </div>
          ))}
        </div>
        <Button className="mt-4" onClick={() => saveMutation.mutate(widgets)}>Save Layout</Button>
      </Card>
    </div>
  )
}