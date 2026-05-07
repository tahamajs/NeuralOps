import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { HardwareProvider } from './features/hardware/context/HardwareContext'
import { MainLayout } from './components/layout/MainLayout'
import FloatingWindowRenderer from './components/floating/FloatingWindow'
import FloatingWindowBar from './components/floating/FloatingWindowBar'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 10000,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <HardwareProvider>
          <MainLayout />
          <FloatingWindowRenderer />
          <FloatingWindowBar />
        </HardwareProvider>
      </BrowserRouter>
    </QueryClientProvider>
  )
}