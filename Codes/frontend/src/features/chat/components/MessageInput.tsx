import { useState, KeyboardEvent } from 'react'
import { Button } from '../../../components/ui/Button'
import { Send } from 'lucide-react'

interface MessageInputProps {
  onSendMessage: (content: string) => void
  disabled?: boolean
}

export function MessageInput({ onSendMessage, disabled }: MessageInputProps) {
  const [input, setInput] = useState('')

  const handleSend = () => {
    if (input.trim() && !disabled) {
      onSendMessage(input.trim())
      setInput('')
    }
  }

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="border-t border-white/10 p-4">
      <div className="flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about your experiments, notes, or anything..."
          className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-white placeholder-slate-500 focus:border-neon-blue/50 resize-none"
          rows={2}
          disabled={disabled}
        />
        <Button onClick={handleSend} disabled={!input.trim() || disabled} className="self-end">
          <Send size={16} />
        </Button>
      </div>
    </div>
  )
}