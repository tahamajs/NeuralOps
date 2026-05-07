

## 📦 Folder Structure – Feature‑Based with Shared Components

```
src/
├── api/                     # API layer
│   ├── client.ts            # Axios instance (already exists)
│   └── endpoints.ts         # Centralised endpoint paths
│
├── assets/                  # Static assets (images, fonts)
│
├── components/              # Shared UI components (Atomic Design)
│   ├── ui/                  # Atoms: Button, Input, Select, Badge, Card, etc.
│   ├── widgets/             # Organisms: ActiveJobsWidget, SystemResourcesWidget…
│   └── layout/              # Layout components: Sidebar, Header, MainLayout
│
├── features/                # Feature modules (each feature is self‑contained)
│   ├── auth/
│   │   ├── components/      # LoginForm, RegisterForm
│   │   ├── pages/           # LoginPage, RegisterPage
│   │   ├── hooks/           # useAuth (login/logout/register)
│   │   ├── store/           # authStore.ts
│   │   └── types.ts
│   │
│   ├── dashboard/
│   │   ├── components/      # MetricSummaryCards, ExperimentTable (if dashboard‑specific)
│   │   ├── pages/           # DashboardPage
│   │   ├── hooks/           # useDashboardData
│   │   └── types.ts
│   │
│   ├── experiments/
│   │   ├── components/      # ExperimentFilters, ExperimentTable
│   │   ├── pages/           # ExperimentListPage, ExperimentDetailPage
│   │   ├── hooks/           # useExperimentWebSocket, useExperimentFilters
│   │   ├── store/           # experimentStore.ts (if needed)
│   │   └── types.ts
│   │
│   ├── launcher/
│   │   ├── components/      # ModelPicker, TaskPicker, HyperparameterFields…
│   │   ├── pages/           # LauncherPage
│   │   ├── hooks/           # useLauncherForm
│   │   └── types.ts
│   │
│   └── hardware/
│       ├── components/      # HardwareTelemetryWidget
│       ├── context/         # HardwareProvider (if truly global)
│       └── hooks.ts         # useHardware
│
├── hooks/                   # Global hooks (useWebSocket, useDebounce)
│   └── useWebSocket.ts
│
├── stores/                  # Global Zustand stores (if any cross‑feature)
│   └── notificationStore.ts
│
├── providers/               # All context providers composed here
│   └── AppProviders.tsx     # Composes QueryClient, HardwareProvider, Router, etc.
│
├── lib/                     # Utilities, constants, helpers
│   ├── constants.ts
│   ├── formatters.ts        # Date/currency formatters
│   └── validators.ts
│
├── styles/
│   ├── globals.css          # Tailwind base + custom layers
│   └── animation.css        # Custom animations
│
├── types/                   # Global TypeScript types
│   ├── api.ts               # API response shapes, request types
│   ├── experiment.ts        # Experiment, TrainingRun interfaces
│   └── hardware.ts          # HardwareMetrics
│
├── App.tsx                  # Routing and layout wrappers
├── main.tsx                 # Entry point
└── router.tsx               # React Router configuration (lazy‑loaded routes)
```

---

## 🧠 Key Architectural Decisions

### 1. **Server State vs. Client State**
- **Server state** (experiments, jobs, configs) → **React Query** (`@tanstack/react-query`)  
  - Automatic caching, background refetch, pagination/infinite scroll.
  - No need to store fetched data in Zustand; React Query is the single source of truth.
- **Client state** (UI toggles, launcher form, auth token) → **Zustand** or **React Context**.
  - Zustand for complex state shared across features (e.g., auth store).
  - Context only for truly app‑wide concerns like `HardwareProvider`.

### 2. **API Layer Abstraction**
- All endpoint URLs are defined in `api/endpoints.ts`, e.g.:
  ```ts
  export const API_ROUTES = {
    AUTH: { LOGIN: '/auth/login/', REGISTER: '/auth/register/' },
    EXPERIMENTS: { LIST: '/experiments/', DETAIL: (id: string) => `/experiments/${id}/` },
    // …
  }
  ```
- The Axios client can be extended with typed request/response generics.

### 3. **Feature‑Based Organization**
- Every feature folder is independent and can be tested in isolation.
- Components that are reused across features move to `components/ui/` or `components/widgets/`.
- Pages are thin orchestrators that compose feature‑specific components.

### 4. **Design System with Tailwind + CVA**
- Define a set of reusable primitive components in `components/ui/` using `class-variance-authority` for variants:
  ```tsx
  // components/ui/Button.tsx
  import { cva, type VariantProps } from 'class-variance-authority'
  const buttonVariants = cva(
    'inline-flex items-center justify-center rounded-xl font-semibold transition-all focus:outline-none',
    {
      variants: {
        variant: {
          primary: 'bg-blue-600 text-white hover:bg-blue-500 shadow-blue-500/20',
          ghost: 'bg-transparent hover:bg-white/5 text-slate-400',
        },
        size: {
          sm: 'px-3 py-1.5 text-xs',
          md: 'px-6 py-2.5 text-sm',
        },
      },
      defaultVariants: { variant: 'primary', size: 'md' },
    }
  )
  ```
- Use consistent glass‑morphism, cyber‑grid background, and color tokens from Tailwind config.

### 5. **Real‑Time WebSocket Layer**
- The `useWebSocket` hook should be generic and reusable, accepting a list of message handlers.
- Store the WebSocket connection status in Zustand so any component can react to disconnections.
- Use React Query’s `queryClient.setQueryData` to push real‑time updates into the cache (e.g., live progress) without refetching.

### 6. **Routing & Code Splitting**
- Use `React.lazy` and `Suspense` for page‑level code splitting (already partially done).
- Create a dedicated `router.tsx` file with route definitions and lazy imports:
  ```tsx
  const DashboardPage = lazy(() => import('@/features/dashboard/pages/DashboardPage'))
  ```
- Wrap routes with a layout component (e.g., `AuthenticatedLayout` containing the top nav and `HardwareProvider`).

### 7. **Error, Loading, and Empty States**
- Every data‑fetching component should handle these three states.
- Create reusable components: `<LoadingSpinner>`, `<ErrorAlert>`, `<EmptyState>`.
- React Query’s `isLoading`, `isError`, `data` make this trivial.

### 8. **Type Safety End‑to‑End**
- Share TypeScript types between backend and frontend via a shared package (or at least define them in `types/api.ts`).
- Use generics for API responses, e.g., `ApiResponse<T>`.
- Leverage Zod (already in dependencies) for form validation and runtime type checking of WebSocket messages.

### 9. **Performance Optimizations**
- **React.memo** for expensive components (e.g., `ExperimentTable`).
- **Virtualization** for long lists (e.g., experiment logs, metrics) – use `@tanstack/react-virtual`.
- Lazy load heavy plotting libraries (Plotly) inside `Suspense`.
- Use `useMemo` and `useCallback` where necessary.

### 10. **Beautiful UI/UX**
- Add smooth page transitions with `framer-motion` or CSS animations.
- Use skeleton loaders (shimmer effect) instead of simple spinners.
- Leverage `lucide-react` icons for consistent iconography.
- Responsive design using Tailwind’s breakpoints; test mobile layout.
- Add a `GlobalNotification` component for success/error toasts (use a Zustand store for notifications).

---

## 🛠️ Refactoring Steps (How to Reach This Architecture)

1. **Clean up duplicate `.js` files** – delete all JavaScript versions and keep only `.tsx`.
2. **Create the folder structure** above.
3. **Move existing pages** into `features/<feature>/pages/`.
4. **Move components**:
   - Generic UI (Button, Input, Select…) → `components/ui/`.
   - Widgets (ActiveJobsWidget, SystemResourcesWidget…) → `components/widgets/`.
5. **Extract a generic `useWebSocket`** and move to `hooks/useWebSocket.ts`.
6. **Define API endpoints** in `api/endpoints.ts` and replace hard‑coded URLs.
7. **Build shared components** (LoadingSpinner, ErrorBoundary, EmptyState).
8. **Enhance the design system** with CVA variants.
9. **Add route‑based code splitting** with `React.lazy`.
10. **Write integration tests** for key user flows (using React Testing Library).

---

## 📊 Visual Architecture Diagram (simplified)

```
┌─────────────────────────────────────────────────────────┐
│                     <AppProviders>                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ QueryClient │  │ HardwareCtx │  │  NotificationCtx│  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│        │                                          │       │
│  ┌─────▼─────┐                              ┌────▼───┐  │
│  │  Router   │                              │  Toast  │  │
│  └─────┬─────┘                              └────────┘  │
│        │                                                  │
│  ┌─────▼─────────────┐                                   │
│  │  Authenticated    │                                   │
│  │  Layout (Nav+Side)│                                   │
│  └─────┬─────────────┘                                   │
│        │                                                  │
│    Page Components (lazy loaded)                          │
│    ┌────────────────────────────┐                         │
│    │  Features (Dashboard,      │                         │
│    │  Experiments, Launcher…)   │                         │
│    └────────────────────────────┘                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔥 Final Result

With this architecture, your application will be:
- **Scalable** – adding a new feature is as simple as creating a new folder under `features/`.
- **Maintainable** – clear separation of concerns; each module is independently testable.
- **Beautiful** – consistent design system and smooth animations.
- **High‑performance** – code splitting, efficient state management, and caching.
- **Real‑time ready** – robust WebSocket layer that integrates seamlessly with React Query.
