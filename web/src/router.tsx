import { createBrowserRouter, Navigate } from 'react-router'

import { AppShell } from './app/AppShell'
import { TeamPageColors } from './app/TeamPageColors'
import { EntityDetailPage } from './pages/EntityDetailPage'
import { EntityIndexPage } from './pages/EntityIndexPage'
import { GlossaryPage } from './pages/GlossaryPage'
import { SeasonDataRoute } from './pages/SeasonDataRoute'

export const routes = [
  {
    path: '/',
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/teams" replace /> },
      {
        path: 'teams',
        element: <SeasonDataRoute kind="teams">{(dataset) => <EntityIndexPage kind="teams" dataset={dataset} />}</SeasonDataRoute>,
      },
      {
        path: 'teams/:entityId',
        element: (
          <TeamPageColors>
            <SeasonDataRoute kind="teams">{(dataset) => <EntityDetailPage kind="teams" dataset={dataset} />}</SeasonDataRoute>
          </TeamPageColors>
        ),
      },
      {
        path: 'qbs',
        element: <SeasonDataRoute kind="qbs">{(dataset) => <EntityIndexPage kind="qbs" dataset={dataset} />}</SeasonDataRoute>,
      },
      {
        path: 'qbs/:entityId',
        element: <SeasonDataRoute kind="qbs">{(dataset) => <EntityDetailPage kind="qbs" dataset={dataset} />}</SeasonDataRoute>,
      },
      { path: 'glossary', element: <GlossaryPage /> },
      { path: '*', element: <Navigate to="/teams" replace /> },
    ],
  },
]

export const router = createBrowserRouter(routes)
