import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "react-router-dom";
import { HardwareProvider } from "../features/hardware/context/HardwareContext";
import FloatingWindowRenderer from "../components/floating/FloatingWindow";
import FloatingWindowBar from "../components/floating/FloatingWindowBar";
import { router } from "../router";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: (failureCount, error: any) => {
        const status = error?.response?.status;
        if (status === 401 || status === 403 || status === 404) return false;
        return failureCount < 2;
      },
      staleTime: 10000,
      gcTime: 5 * 60 * 1000,
    },
    mutations: {
      retry: false,
    },
  },
});

export default function AppProviders() {
  return (
    <QueryClientProvider client={queryClient}>
      <HardwareProvider>
        <RouterProvider router={router} />
        <FloatingWindowRenderer />
        <FloatingWindowBar />
      </HardwareProvider>
    </QueryClientProvider>
  );
}
