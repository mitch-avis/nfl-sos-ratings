import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError, apiFetch, apiPost } from './client'

afterEach(() => {
  vi.unstubAllGlobals()
})

function respond(body: string, init: ResponseInit) {
  vi.stubGlobal('fetch', async () => new Response(body, init))
}

describe('apiFetch', () => {
  it('returns parsed JSON', async () => {
    // Arrange
    respond('{"seasons":[2025]}', { status: 200, headers: { 'content-type': 'application/json' } })

    // Act & Assert
    await expect(apiFetch('/api/seasons')).resolves.toEqual({ seasons: [2025] })
  })

  it("surfaces the backend's detail message", async () => {
    // Arrange
    respond('{"detail":"Season 1990 is not available"}', {
      status: 404,
      headers: { 'content-type': 'application/json' },
    })

    // Act & Assert
    await expect(apiFetch('/api/seasons/1990')).rejects.toEqual(new ApiError(404, 'Season 1990 is not available'))
  })

  it('explains an HTML response from a server that is not the API', async () => {
    // Arrange
    respond('<!doctype html>', { status: 200, headers: { 'content-type': 'text/html' } })

    // Act & Assert
    await expect(apiFetch('/api/seasons')).rejects.toThrow(/is nfl-sos-ratings web running/)
  })
})

describe('apiPost', () => {
  it("posts with the app's header and returns parsed JSON", async () => {
    // Arrange
    const fetchSpy = vi.fn(async () => new Response('{"state":"running"}', { status: 202, headers: { 'content-type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchSpy)

    // Act
    const body = await apiPost('/api/refresh')

    // Assert
    expect(body).toEqual({ state: 'running' })
    expect(fetchSpy).toHaveBeenCalledWith('/api/refresh', { method: 'POST', headers: { 'X-Requested-With': 'nfl-sos-ratings' } })
  })

  it("surfaces the backend's detail message", async () => {
    // Arrange
    respond('{"detail":"A refresh is already running."}', { status: 409, headers: { 'content-type': 'application/json' } })

    // Act & Assert
    await expect(apiPost('/api/refresh')).rejects.toEqual(new ApiError(409, 'A refresh is already running.'))
  })
})
