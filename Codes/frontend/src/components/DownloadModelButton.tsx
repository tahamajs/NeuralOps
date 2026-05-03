export default function DownloadModelButton({ experimentId }: { experimentId: string }) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-2">Model</h2>
      <a
        href={`/api/experiments/${experimentId}/model/`}
        download
        className="inline-block bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        Download Model (ZIP)
      </a>
    </div>
  )
}