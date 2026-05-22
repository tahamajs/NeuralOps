import { marked } from 'marked'
import DOMPurify from 'dompurify'

marked.setOptions({
  breaks: true,
  gfm: true,
})

export function MarkdownRenderer({ content, className }: { content: string; className?: string }) {
  const html = marked(content)
  const sanitized = DOMPurify.sanitize(html)

  return (
    <div
      className={cn('prose prose-invert max-w-none', className)}
      dangerouslySetInnerHTML={{ __html: sanitized }}
    />
  )
}