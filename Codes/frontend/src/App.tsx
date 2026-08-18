// src/App.tsx
import { RouterProvider } from "react-router-dom";
import { router } from "./router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "next-themes";
import { HardwareProvider } from "./features/hardware/context/HardwareContext";

// ========================
// QueryClient
// ========================
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 10000,
    },
  },
});

// ========================
// Main App Component
// ========================
function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <HardwareProvider>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
          <RouterProvider router={router} />
        </ThemeProvider>
      </HardwareProvider>
    </QueryClientProvider>
  );
}

export default App;
