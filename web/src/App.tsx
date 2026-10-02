import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from 'react-router'

import { EntityViewStateProvider } from './app/EntityViewStateProvider'
import { ThemeProvider } from './app/ThemeProvider'
import { TooltipProvider } from './components/ui/tooltip'
import { router } from './router'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 5 * 60_000, refetchOnWindowFocus: false, retry: 1 },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider delayDuration={200}>
          <EntityViewStateProvider>
            <RouterProvider router={router} />
          </EntityViewStateProvider>
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  )
}
