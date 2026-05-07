import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Image from '@tiptap/extension-image'
import Table from '@tiptap/extension-table'
import TableRow from '@tiptap/extension-table-row'
import TableCell from '@tiptap/extension-table-cell'
import TableHeader from '@tiptap/extension-table-header'
import CodeBlock from '@tiptap/extension-code-block'
import Placeholder from '@tiptap/extension-placeholder'
import { Button } from '../../../components/ui/Button'
import { Bold, Italic, List, ListOrdered, Code, Quote, ImageIcon, Table2 } from 'lucide-react'

interface RichNoteEditorProps {
  content: string
  onChange: (html: string) => void
}

export default function RichNoteEditor({ content, onChange }: RichNoteEditorProps) {
  const editor = useEditor({
    extensions: [
      StarterKit,
      Image,
      Table.configure({ resizable: true }),
      TableRow, TableCell, TableHeader,
      CodeBlock,
      Placeholder.configure({ placeholder: 'Write something amazing...' }),
    ],
    content,
    onUpdate: ({ editor }) => {
      onChange(editor.getHTML())
    },
  })

  if (!editor) return null

  return (
    <div className="glass border border-white/10 rounded-2xl overflow-hidden">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-1 p-2 border-b border-white/5 bg-cyber-800">
        <button onClick={() => editor.chain().focus().toggleBold().run()} className={`p-2 rounded hover:bg-white/10 ${editor.isActive('bold') ? 'bg-neon-blue/20 text-neon-blue' : 'text-slate-400'}`}><Bold size={16} /></button>
        <button onClick={() => editor.chain().focus().toggleItalic().run()} className={`p-2 rounded hover:bg-white/10 ${editor.isActive('italic') ? 'bg-neon-blue/20 text-neon-blue' : 'text-slate-400'}`}><Italic size={16} /></button>
        <button onClick={() => editor.chain().focus().toggleBulletList().run()} className={`p-2 rounded hover:bg-white/10 ${editor.isActive('bulletList') ? 'bg-neon-blue/20 text-neon-blue' : 'text-slate-400'}`}><List size={16} /></button>
        <button onClick={() => editor.chain().focus().toggleOrderedList().run()} className={`p-2 rounded hover:bg-white/10 ${editor.isActive('orderedList') ? 'bg-neon-blue/20 text-neon-blue' : 'text-slate-400'}`}><ListOrdered size={16} /></button>
        <button onClick={() => editor.chain().focus().toggleCodeBlock().run()} className={`p-2 rounded hover:bg-white/10 ${editor.isActive('codeBlock') ? 'bg-neon-blue/20 text-neon-blue' : 'text-slate-400'}`}><Code size={16} /></button>
        <button onClick={() => editor.chain().focus().toggleBlockquote().run()} className={`p-2 rounded hover:bg-white/10 ${editor.isActive('blockquote') ? 'bg-neon-blue/20 text-neon-blue' : 'text-slate-400'}`}><Quote size={16} /></button>
        <button onClick={() => editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run()} className="p-2 rounded hover:bg-white/10 text-slate-400"><Table2 size={16} /></button>
        <button onClick={() => {
          const url = window.prompt('Image URL')
          if (url) editor.chain().focus().setImage({ src: url }).run()
        }} className="p-2 rounded hover:bg-white/10 text-slate-400"><ImageIcon size={16} /></button>
      </div>
      {/* Editor */}
      <EditorContent editor={editor} className="prose prose-invert max-w-none p-4 min-h-[300px] focus:outline-none text-white" />
    </div>
  )
}