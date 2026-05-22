import { useEffect, useRef } from 'react'
import { fabric } from 'fabric'

export function BoundingBoxCanvas({ imageUrl, onAnnotationsChange, labels }: any) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fabricRef = useRef<fabric.Canvas | null>(null)

  useEffect(() => {
    if (!canvasRef.current) return
    const canvas = new fabric.Canvas(canvasRef.current)
    fabricRef.current = canvas
    fabric.Image.fromURL(imageUrl, img => {
      const scale = Math.min(canvas.width! / img.width!, canvas.height! / img.height!)
      img.scale(scale).set({ left: (canvas.width! - img.width! * scale) / 2, top: (canvas.height! - img.height! * scale) / 2 })
      canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas))
    })
    canvas.on('object:added', () => onAnnotationsChange(canvas.getObjects().filter(o => o.type === 'rect').map(r => ({ label_name: 'Object', geometry: r.toJSON() }))))
    return () => canvas.dispose()
  }, [imageUrl])

  return <canvas ref={canvasRef} width={800} height={600} className="border border-white/10 rounded" />
}