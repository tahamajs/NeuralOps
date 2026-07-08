import { marked } from "marked";
import DOMPurify from "dompurify";
import { cn } from "../../../../lib/utils";

marked.setOptions({
  breaks: true,
  gfm: true,
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
});

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export function MarkdownRenderer({
  content,
  className,
}: MarkdownRendererProps) {
  const html = marked(content);
  const sanitized = DOMPurify.sanitize(html, {
    ADD_TAGS: ["iframe"],
    ADD_ATTR: ["target", "allow", "allowfullscreen"],
  });

  return (
    <div
      className={cn("prose prose-invert max-w-none", className)}
      dangerouslySetInnerHTML={{ __html: sanitized }}
    />
  );
}
