import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError, apiFetch } from './client'

afterEach(() => {
  vi.unstubAllGlobals()
})

function respond(body: string, init: ResponseInit) {
  vi.stubGlobal('fetch', async () => new Response(body, init))
}

describe('apiFetch', () => {
  it('returns parsed JSON', async () => {
    respond('{"seasons":[2025]}', { status: 200, headers: { 'content-type': 'application/json' } })
    await expect(apiFetch('/api/seasons')).resolves.toEqual({ seasons: [2025] })
  })

  it("surfaces the backend's detail message", async () => {
    respond('{"detail":"Season 1990 is not available"}', {
      status: 404,
      headers: { 'content-type': 'application/json' },
    })
    await expect(apiFetch('/api/seasons/1990')).rejects.toEqual(new ApiError(404, 'Season 1990 is not available'))
  })

  it('explains an HTML response from a server that is not the API', async () => {
    respond('<!doctype html>', { status: 200, headers: { 'content-type': 'text/html' } })
    await expect(apiFetch('/api/seasons')).rejects.toThrow(/is nfl-sos-ratings web running/)
  })
})
