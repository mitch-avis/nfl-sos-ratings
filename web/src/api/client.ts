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
  const response = await fetch(path, { signal })
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
