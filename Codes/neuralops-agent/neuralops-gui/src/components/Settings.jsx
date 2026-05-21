import React, { useState, useEffect } from 'react';
import { Save, RefreshCw } from 'lucide-react';
import { api } from '../services/api';

export default function Settings() {
  const [listings, setListings] = useState([]);
  const [config, setConfig] = useState({ base_price_per_hour: '0.50', max_concurrent_jobs: 2, storage_gb: 100 });

  useEffect(() => {
    api.getMyListings().then(data => {
      if (data.length > 0) {
        const l = data[0];
        setConfig({
          base_price_per_hour: l.base_price_per_hour,
          max_concurrent_jobs: l.max_rental_hours || 2,
          storage_gb: l.storage_gb || 100,
          listing_id: l.id,
        });
        setListings(data);
      }
    });
  }, []);

  const handleSave = async () => {
    if (config.listing_id) {
      await api.updateListing(config.listing_id, {
        base_price_per_hour: config.base_price_per_hour,
        storage_gb: config.storage_gb,
      });
      alert('Settings saved!');
    }
  };

  return (
    <div className="space-y-6 max-w-2xl">
      <h2 className="text-2xl font-bold">Settings</h2>
      <div className="bg-dark-card border border-dark-border rounded-xl p-6 space-y-4">
        <div>
          <label className="text-gray-400 text-sm">Base Price per Hour (USD)</label>
          <input type="number" step="0.01" value={config.base_price_per_hour}
            onChange={e => setConfig({...config, base_price_per_hour: e.target.value})}
            className="w-full mt-1 bg-dark-bg border border-dark-border rounded-lg px-3 py-2 text-white" />
        </div>
        <div>
          <label className="text-gray-400 text-sm">Storage Limit (GB)</label>
          <input type="number" value={config.storage_gb}
            onChange={e => setConfig({...config, storage_gb: parseInt(e.target.value)})}
            className="w-full mt-1 bg-dark-bg border border-dark-border rounded-lg px-3 py-2 text-white" />
        </div>
        <button onClick={handleSave}
          className="flex items-center gap-2 bg-primary-600 hover:bg-primary-700 px-4 py-2 rounded-lg font-semibold transition-colors">
          <Save size={18} /> Save Settings
        </button>
      </div>
    </div>
  );
}