export default function LaunchButton({ loading, onClick }: { loading: boolean; onClick: () => void }) {
  return (
    <button
      disabled={loading}
      onClick={onClick}
      className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 disabled:opacity-50 text-lg font-semibold w-full"
    >
      {loading ? 'Launching...' : '🚀 Launch Experiment'}
    </button>
  )
}