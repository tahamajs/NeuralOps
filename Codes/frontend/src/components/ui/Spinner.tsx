export const Spinner = ({ className }: { className?: string }) => (
  <div className={`flex items-center justify-center py-12 ${className}`}>
    <div className="w-8 h-8 border-2 border-white/10 border-t-neon-blue rounded-full animate-spin" />
  </div>
)