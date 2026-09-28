/**
 * Where the backend lives.
 *
 * Set VITE_API_URL at build time to point a deployed frontend at a deployed
 * API. Vite inlines env vars at build time, so this is baked into the bundle
 * rather than read at runtime - changing it means rebuilding, which on Vercel
 * means a redeploy after editing the environment variable.
 *
 * The fallback is the local dev server, so a checkout works with no setup.
 */
export const API = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'
