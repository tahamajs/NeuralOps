import { useEffect, useRef, forwardRef, useImperativeHandle } from 'react'
import SignaturePad from 'signature_pad'

interface SignatureCanvasProps {
  onSave: (data: string) => void
}

export const SignatureCanvas = forwardRef(({ onSave }: SignatureCanvasProps, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const padRef = useRef<SignaturePad | null>(null)

  useEffect(() => {
    if (canvasRef.current) {
      padRef.current = new SignaturePad(canvasRef.current)
    }
  }, [])

  const handleSave = () => {
    if (padRef.current && !padRef.current.isEmpty()) {
      onSave(padRef.current.toDataURL())
    }
  }

  const handleClear = () => {
    padRef.current?.clear()
  }

  useImperativeHandle(ref, () => ({ clear: handleClear, save: handleSave }))

  return (
    <div className="space-y-2"><canvas ref={canvasRef} width={500} height={200} className="border border-white/10 rounded bg-white" style={{ background: 'white' }} /><div className="flex gap-2"><button onClick={handleClear} className="px-3 py-1 text-sm bg-white/10 rounded">Clear</button><button onClick={handleSave} className="px-3 py-1 text-sm bg-neon-blue rounded">Save Signature</button></div></div>
  )
})