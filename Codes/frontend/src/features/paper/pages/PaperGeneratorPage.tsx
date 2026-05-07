import { useParams } from 'react-router-dom'
import { useState } from 'react'
import api from '../../../api/client'
import { Button } from '../../../components/ui/Button'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { useQuery } from '@tanstack/react-query'

export default function PaperGeneratorPage() {
  const { runId } = useParams<{ runId: string }>()
  const [loading, setLoading] = useState(false)

  const downloadPaper = async () => {
    setLoading(true)
    try {
      const response = await fetch(`/api/runs/${runId}/paper/`)
      if (!response.ok) throw new Error('Generation failed')
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      const ext = response.headers.get('content-type')?.includes('pdf') ? 'pdf' : 'tex'
      link.download = `paper_${runId}.${ext}`
      link.click()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      alert('Paper generation failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <h1 className="text-2xl font-black text-white">Paper Generation</h1>
      <Card>
        <p className="text-slate-400 mb-4">Generate a LaTeX paper from this experiment's results.</p>
        <Button onClick={downloadPaper} disabled={loading}>
          {loading ? 'Generating...' : '📄 Generate Paper'}
        </Button>
        <p className="text-[10px] text-slate-500 mt-2">
          A .tex file will be downloaded. Compile with pdflatex to get a PDF.
        </p>
      </Card>
    </div>
  )
}