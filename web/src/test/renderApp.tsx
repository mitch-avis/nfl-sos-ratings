import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router'

import { EntityViewStateProvider } from '@/app/EntityViewStateProvider'
import { ThemeProvider } from '@/app/ThemeProvider'
import { TooltipProvider } from '@/components/ui/tooltip'
import { routes } from '@/router'

/** Render the whole app at `path` with a fresh query cache and in-memory router. */
export function renderApp(path: string) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter(routes, { initialEntries: [path] })
  const result = render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider delayDuration={0}>
          <EntityViewStateProvider>
            <RouterProvider router={router} />
          </EntityViewStateProvider>
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>,
  )
  return { ...result, router }
}
