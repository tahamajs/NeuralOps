import { useState } from 'react'
import { Input } from '../../../components/ui/Input'
import { Button } from '../../../components/ui/Button'
import { Send, Paperclip } from 'lucide-react'

interface Props {
  onSend: (text: string) => void
  disabled: boolean
}

export default function ChatInput({ onSend, disabled }: Props) {
  const [input, setInput] = useState('')

  const handleSend = () => {
    if (!input.trim()) return
    onSend(input)
    setInput('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="glass p-3 flex gap-2">
      <Input
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask me anything – search papers, configure experiments, analyze training..."
        className="flex-1"
        disabled={disabled}
      />
      <Button onClick={handleSend} disabled={disabled || !input.trim()}>
        <Send size={16} />
      </Button>
    </div>
  )
}