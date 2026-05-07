import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from 'react-router-dom'
import { HardwareProvider } from '../features/hardware/context/HardwareContext'
import FloatingWindowRenderer from '../components/floating/FloatingWindow'
import FloatingWindowBar from '../components/floating/FloatingWindowBar'
import { router } from '../router'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 10_000,
    },
  },
})

export default function AppProviders() {
  return (
    <QueryClientProvider client={queryClient}>
      <HardwareProvider>
        <RouterProvider router={router} />
        {/* Persistent floating windows – available everywhere */}
        <FloatingWindowRenderer />
        <FloatingWindowBar />
      </HardwareProvider>
    </QueryClientProvider>
  )
}