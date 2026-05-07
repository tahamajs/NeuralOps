import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Switch } from '../../../components/ui/Switch'
import { Spinner } from '../../../components/ui/Spinner'
import { Settings } from 'lucide-react'

export default function DashboardWidgetsConfigPage() {
  const queryClient = useQueryClient()

  const { data: widgets = [], isLoading } = useQuery({
    queryKey: ['dashboardWidgets'],
    queryFn: async () => (await api.get('/api/widgets/')).data,
  })

  const saveMutation = useMutation({
    mutationFn: async (layout: any) =>
      api.put('/api/widgets/layout/', { widgets: layout }),
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
    <div className="space-y-6 animate-fade-in max-w-2xl mx-auto">
      <div className="flex items-center gap-3">
        <Settings size={24} className="text-neon-blue" />
        <h1 className="text-2xl font-display font-bold text-white">Dashboard Widgets</h1>
      </div>

      <Card>
        <p className="text-sm text-slate-400 mb-6">
          Choose which widgets appear on your main dashboard.
        </p>

        {widgets.length === 0 ? (
          <p className="text-slate-500 text-sm">No widgets configured yet.</p>
        ) : (
          <div className="space-y-3">
            {widgets.map((w: any) => (
              <div
                key={w.id}
                className="flex items-center justify-between p-4 bg-cyber-800 rounded-xl border border-white/5"
              >
                <div>
                  <p className="text-white font-semibold text-sm">{w.title || w.widget_type}</p>
                  <p className="text-[10px] text-slate-400 uppercase tracking-wider mt-0.5">
                    {w.widget_type}
                  </p>
                </div>
                <Switch checked={w.is_visible} onChange={() => handleToggle(w.id)} />
              </div>
            ))}
          </div>
        )}

        <Button
          className="mt-6"
          onClick={() => saveMutation.mutate(widgets)}
          loading={saveMutation.isPending}
        >
          Save Layout
        </Button>
      </Card>
    </div>
  )
}