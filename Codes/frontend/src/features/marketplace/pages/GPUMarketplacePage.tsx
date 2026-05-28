// src/features/marketplace/pages/GPUMarketplacePage.tsx
import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../../../api/client'
import { useAuthStore } from '../../../features/auth/store/authStore'
import { Card, Button, Input, Spinner, Select, Badge } from '../../../components/ui'
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, Legend 
} from 'recharts'

// --- Types ---
interface GPUMarketplaceListing {
  id: string
  name?: string
  provider_username: string
  gpu_model: string
  gpu_count: number
  memory_gb?: number
  vram_gb?: number
  price_per_hour?: number
  current_price_per_hour?: number
  status: string
  region?: string
  location?: string
  created_at: string
  reliability_score: number
  performance_score?: number
  availability_percentage?: number
  uptime_percentage?: number
  verified?: boolean
  verification_level?: string
}

interface MarketDepth {
  model: string
  total_available: number
  avg_price: number
  min_price: number
  max_price: number
  sellers_count: number
}

// --- API service (using the global api instance) ---
const marketplaceAPI = {
  getListings: (params?: any) =>
    api.get('/marketplace/listings/', { params }).then(res => res.data),
  getHistoricalPrices: (listingId: string, days = 30) =>
    api.get(`/marketplace/listings/${listingId}/history/`, { params: { days } }).then(res => res.data),
  getMarketDepth: () =>
    api.get('/marketplace/depth/').then(res => res.data),
  getWatchlist: () =>
    api.get('/marketplace/watchlist/').then(res => res.data),
  addToWatchlist: (listingId: string) =>
    api.post('/marketplace/watchlist/', { listing_id: listingId }).then(res => res.data),
  removeFromWatchlist: (listingId: string) =>
    api.delete(`/marketplace/watchlist/${listingId}/`).then(res => res.data),
}

// --- Listing Card Component ---
const ListingCard = ({ listing, onSelect, onAddToWatchlist }: any) => {
  return (
    <Card hover className="p-3 cursor-pointer" onClick={() => onSelect(listing.id)}>
      <div className="flex justify-between items-center">
        <div>
          <div className="font-medium text-white">{listing.name}</div>
          <div className="text-xs text-slate-400">{listing.provider_username} · {listing.gpu_model} x{listing.gpu_count}</div>
          <div className="flex flex-wrap gap-1 mt-1">
            <Badge variant="info">{listing.region || listing.location || 'Unknown'}</Badge>
            {listing.verified && <Badge variant="success">Verified</Badge>}
          </div>
        </div>
        <div className="text-right">
          <div className="text-neon-blue font-bold">${(listing.price_per_hour || listing.current_price_per_hour || 0).toFixed(4)} USD/hr</div>
          <div className="text-[10px] text-green-400">{(listing.availability_percentage || listing.uptime_percentage || 0)}% avail</div>
          <Button variant="ghost" size="sm" onClick={(e) => { e.stopPropagation(); onAddToWatchlist(); }} className="mt-1">
            Add to Watchlist
          </Button>
        </div>
      </div>
    </Card>
  )
}

// --- Filter Bar ---
const FilterBar = ({ filters, setFilters, gpuModels, regions }: any) => {
  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 flex flex-wrap gap-3 items-end">
      <Input label="Search" value={filters.search || ''} onChange={(e) => setFilters({ ...filters, search: e.target.value })} placeholder="GPU, provider, ID..." className="flex-1 min-w-[150px]" />
      <Select label="GPU Model" options={[{ value: '', label: 'All Models' }, ...gpuModels.map((m: string) => ({ value: m, label: m }))]} value={filters.gpu_model} onChange={(e) => setFilters({ ...filters, gpu_model: e.target.value })} className="w-48" />
      <Select label="Region" options={[{ value: '', label: 'All Regions' }, ...regions.map((r: string) => ({ value: r, label: r }))]} value={filters.region} onChange={(e) => setFilters({ ...filters, region: e.target.value })} className="w-36" />
      <div className="flex gap-2">
        <Input type="number" label="Min Price" value={filters.min_price || ''} onChange={(e) => setFilters({ ...filters, min_price: e.target.value ? parseFloat(e.target.value) : '' })} placeholder="Min USD/hr" className="w-32" />
        <Input type="number" label="Max Price" value={filters.max_price || ''} onChange={(e) => setFilters({ ...filters, max_price: e.target.value ? parseFloat(e.target.value) : '' })} placeholder="Max USD/hr" className="w-32" />
      </div>
      <div className="flex gap-2">
        <label className="flex items-center gap-1 text-sm text-white">
          <input type="checkbox" checked={filters.verified} onChange={(e) => setFilters({ ...filters, verified: e.target.checked })} /> Verified only
        </label>
        <Button variant="secondary" size="sm" onClick={() => setFilters({ search: '', region: '', gpu_model: '', min_price: '', max_price: '', verified: false })}>Clear Filters</Button>
      </div>
    </div>
  )
}

// --- Market Depth View ---
const MarketDepthView = ({ marketDepth }: { marketDepth: MarketDepth[] }) => {
  if (!marketDepth || marketDepth.length === 0) return <div className="text-slate-500 text-sm text-center py-4">No market depth data available.</div>
  const totalAvailable = marketDepth.reduce((sum, m) => sum + m.total_available, 0)
  return (
    <div className="space-y-3">
      <div className="flex justify-between text-sm text-slate-400 border-b border-white/10 pb-2">
        <span>GPU Model</span> <span>Total Available</span> <span>Avg Price (USD/hr)</span> <span>Min - Max</span>
      </div>
      <div className="space-y-2 max-h-80 overflow-y-auto">
        {marketDepth.map((depth) => (
          <div key={depth.model} className="flex justify-between text-sm items-center p-2 rounded hover:bg-white/5">
            <span className="font-mono text-white">{depth.model}</span>
            <span className="text-slate-300">{depth.total_available}</span>
            <span className="text-neon-blue">${depth.avg_price.toFixed(4)}</span>
            <span className="text-slate-500 text-xs">${depth.min_price.toFixed(4)} - ${depth.max_price.toFixed(4)}</span>
          </div>
        ))}
      </div>
      <div className="text-sm text-slate-400 pt-2 border-t border-white/10 text-center">Total available compute units: {totalAvailable.toLocaleString()}</div>
    </div>
  )
}

// --- Listing Detail Modal ---
const ListingDetailModal = ({ listing, onClose }: { listing: GPUMarketplaceListing; onClose: () => void }) => {
  const { data: priceHistory, isLoading: historyLoading } = useQuery({
    queryKey: ['historical-prices', listing.id],
    queryFn: () => marketplaceAPI.getHistoricalPrices(listing.id, 30),
    enabled: !!listing.id,
  })
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm" onClick={onClose}>
      <div className="glass p-6 rounded-2xl w-full max-w-3xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-white">{listing.name}</h2>
          <Button variant="ghost" onClick={onClose}>✕</Button>
        </div>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div><span className="text-slate-400">Provider</span><div>{listing.provider_username}</div></div>
            <div><span className="text-slate-400">GPU Model</span><div>{listing.gpu_model} x{listing.gpu_count}</div></div>
            <div><span className="text-slate-400">Memory</span><div>{listing.memory_gb} GB</div></div>
            <div><span className="text-slate-400">Region</span><div>{listing.region}</div></div>
          </div>
          <div className="p-4 bg-white/5 rounded-xl">
            <h3 className="text-sm font-bold text-white mb-2">Price History (30 days)</h3>
            {historyLoading ? <Spinner /> : (priceHistory && priceHistory.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={priceHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                  <XAxis dataKey="timestamp" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#94a3b8' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Legend />
                  <Line type="monotone" dataKey="price_per_hour" stroke="#3b82f6" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-slate-500 text-center py-8">No price history available</div>
            ))}
          </div>
          <div className="flex gap-3 pt-3">
            <Button variant="primary" className="flex-1">Rent Now</Button>
            <Button variant="ghost" className="flex-1">Add to Watchlist</Button>
          </div>
        </div>
      </div>
    </div>
  )
}

// --- Main Page Component ---
export default function GPUMarketplacePage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [selectedListingId, setSelectedListingId] = useState<string | null>(null)
  const [filters, setFilters] = useState({
    search: searchParams.get('q') || '',
    region: searchParams.get('region') || '',
    gpu_model: searchParams.get('model') || '',
    min_price: searchParams.get('min') ? parseFloat(searchParams.get('min')!) : '',
    max_price: searchParams.get('max') ? parseFloat(searchParams.get('max')!) : '',
    verified: searchParams.get('verified') === 'true',
  })
  const [viewMode, setViewMode] = useState<'list' | 'depth'>('list')
  const queryClient = useQueryClient()
  const { user } = useAuthStore()

  const { data: listingsData, isLoading: listingsLoading } = useQuery({
    queryKey: ['marketplace-listings', filters],
    queryFn: () => marketplaceAPI.getListings(filters),
  })
  const { data: marketDepth, isLoading: depthLoading } = useQuery({
    queryKey: ['market-depth'],
    queryFn: () => marketplaceAPI.getMarketDepth(),
  })
  const { data: watchlist, isLoading: watchlistLoading } = useQuery({
    queryKey: ['watchlist'],
    queryFn: () => marketplaceAPI.getWatchlist(),
    enabled: !!user,
  })

  const listings = (listingsData?.results || []).map((listing: GPUMarketplaceListing) => ({
    ...listing,
    name: listing.name || `${listing.gpu_model} x${listing.gpu_count}`,
    memory_gb: listing.memory_gb ?? listing.vram_gb ?? 0,
    price_per_hour: listing.price_per_hour ?? listing.current_price_per_hour ?? 0,
    region: listing.region || listing.location || 'Unknown',
    availability_percentage: listing.availability_percentage ?? listing.uptime_percentage ?? 0,
    verified: listing.verified ?? listing.verification_level === 'enterprise' || listing.verification_level === 'basic',
  }))
  const gpuModels = [...new Set(listings.map((l: GPUMarketplaceListing) => l.gpu_model))]
  const regions = [...new Set(listings.map((l: GPUMarketplaceListing) => l.region))]

  const handleFilterChange = (newFilters: typeof filters) => {
    setFilters(newFilters)
    const params = new URLSearchParams()
    if (newFilters.search) params.set('q', newFilters.search)
    if (newFilters.region) params.set('region', newFilters.region)
    if (newFilters.gpu_model) params.set('model', newFilters.gpu_model)
    if (newFilters.min_price) params.set('min', newFilters.min_price.toString())
    if (newFilters.max_price) params.set('max', newFilters.max_price.toString())
    if (newFilters.verified) params.set('verified', 'true')
    setSearchParams(params)
  }

  const addToWatchlist = (id: string) => marketplaceAPI.addToWatchlist(id).then(() => queryClient.invalidateQueries({ queryKey: ['watchlist'] }))
  const removeFromWatchlist = (id: string) => marketplaceAPI.removeFromWatchlist(id).then(() => queryClient.invalidateQueries({ queryKey: ['watchlist'] }))

  const selectedListing = listings.find((l: GPUMarketplaceListing) => l.id === selectedListingId)
  // No need for `filteredListings` – we already pass filters to API

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-white">GPU Marketplace</h1>
        <div className="flex gap-2">
          <Button variant={viewMode === 'list' ? 'primary' : 'ghost'} size="sm" onClick={() => setViewMode('list')}>Market View</Button>
          <Button variant={viewMode === 'depth' ? 'primary' : 'ghost'} size="sm" onClick={() => setViewMode('depth')}>📊 Depth Chart</Button>
        </div>
      </div>

      <FilterBar filters={filters} setFilters={handleFilterChange} gpuModels={gpuModels} regions={regions} />

      {viewMode === 'depth' ? (
        <Card><div className="p-4"><MarketDepthView marketDepth={marketDepth || []} /></div></Card>
      ) : (
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Card><div className="p-4 space-y-3">
              {listingsLoading && <Spinner />}
              {!listingsLoading && listings.length === 0 && <div className="text-center py-12 text-slate-500">No listings match your filters.</div>}
              {listings.map((listing: GPUMarketplaceListing) => (
                <ListingCard key={listing.id} listing={listing} onSelect={() => setSelectedListingId(listing.id)} onAddToWatchlist={() => addToWatchlist(listing.id)} />
              ))}
            </div></Card>
          </div>
          <div>
            <Card><div className="p-4">
              <h3 className="text-sm font-bold text-white mb-2">Watchlist</h3>
              {watchlistLoading ? <Spinner /> : watchlist?.length ? (
                <div className="space-y-2">
                  {watchlist.map((item: GPUMarketplaceListing) => (
                    <div key={item.id} className="flex justify-between items-center p-2 rounded bg-white/5">
                      <div className="text-sm text-white">{item.name}</div>
                      <Button size="sm" variant="ghost" onClick={() => removeFromWatchlist(item.id)}>Remove</Button>
                    </div>
                  ))}
                </div>
              ) : <p className="text-slate-500 text-sm">No watchlist items.</p>}
            </div></Card>
          </div>
        </div>
      )}

      {selectedListing && <ListingDetailModal listing={selectedListing} onClose={() => setSelectedListingId(null)} />}
    </div>
  )
}