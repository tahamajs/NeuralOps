export default function AdminUsersPage() {
  return (
    <div className="animate-fade-in text-center py-16">
      <h1 className="text-2xl font-display font-bold text-white mb-4">User Management</h1>
      <p className="text-slate-400 mb-6">Use the Django Admin for full user management.</p>
      <a href="/admin/auth/user/" target="_blank" className="inline-block bg-neon-blue text-white px-6 py-3 rounded-xl font-bold">
        Open Django Admin
      </a>
    </div>
  )
}