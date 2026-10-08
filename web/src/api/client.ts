/**
 * Thin fetch wrapper for the FastAPI backend (`nfl-sos-ratings web`).
 *
 * Errors are normalized to `ApiError` so pages can show the backend's message.
 */

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  let detail: unknown = null
  try {
    detail = ((await response.json()) as { detail?: unknown }).detail
  } catch {
    detail = null
  }
  const message =
    typeof detail === 'string' ? detail : `${response.status} ${response.statusText}`.trim()
  return new ApiError(response.status, message)
}

/** GET a JSON payload from `/api/...`. */
export async function apiFetch<T>(path: string, signal?: AbortSignal): Promise<T> {
  return readJson<T>(await fetch(path, { signal }), path)
}

/**
 * POST to `/api/...` with the header the server asks of the app's own requests (a page on another
 * site cannot send it without the server's consent), and return the JSON reply.
 */
export async function apiPost<T>(path: string): Promise<T> {
  return readJson<T>(await fetch(path, { method: 'POST', headers: { 'X-Requested-With': 'nfl-sos-ratings' } }), path)
}

async function readJson<T>(response: Response, path: string): Promise<T> {
  if (!response.ok) {
    throw await toApiError(response)
  }
  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.includes('application/json')) {
    throw new ApiError(
      response.status,
      `The API returned ${contentType || 'no content type'} for ${path}; is nfl-sos-ratings web running?`,
    )
  }
  return (await response.json()) as T
}
