/**
 * Rate limiter — sliding window per-IP tracking
 * 
 * In-memory rate limiting with configurable limits per route prefix.
 * Automatically cleans up stale entries every 5 minutes.
 */

const DEFAULT_LIMITS = {
  '/api/recommendations': { maxRequests: 30, windowMs: 60_000 },
  '/api/sync':            { maxRequests: 2,  windowMs: 60_000 },
  '/api/':                { maxRequests: 60, windowMs: 60_000 },
}

// Map<ip, Map<route, timestamp[]>>
const requestLog = new Map()

// Cleanup interval — remove entries older than 5 minutes
const CLEANUP_INTERVAL = 5 * 60_000
let cleanupTimer = null

function startCleanup() {
  if (cleanupTimer) return
  cleanupTimer = setInterval(() => {
    const cutoff = Date.now() - 5 * 60_000
    for (const [ip, routes] of requestLog) {
      for (const [route, timestamps] of routes) {
        const filtered = timestamps.filter(t => t > cutoff)
        if (filtered.length === 0) {
          routes.delete(route)
        } else {
          routes.set(route, filtered)
        }
      }
      if (routes.size === 0) {
        requestLog.delete(ip)
      }
    }
  }, CLEANUP_INTERVAL)
  // Allow process to exit gracefully
  if (cleanupTimer.unref) cleanupTimer.unref()
}

/**
 * Find the matching limit config for a given path.
 * Checks specific routes first, then falls back to general prefix.
 */
function findLimit(pathname, customLimits) {
  const limits = customLimits || DEFAULT_LIMITS
  
  // Check exact match first
  if (limits[pathname]) return limits[pathname]
  
  // Check prefix match (longest prefix wins)
  let bestMatch = null
  let bestLen = 0
  for (const [prefix, config] of Object.entries(limits)) {
    if (pathname.startsWith(prefix) && prefix.length > bestLen) {
      bestMatch = config
      bestLen = prefix.length
    }
  }
  
  return bestMatch
}

/**
 * Check if a request should be rate limited.
 * 
 * @param {string} ip - Client IP address
 * @param {string} pathname - Request path
 * @param {object} [customLimits] - Optional custom limit config
 * @returns {{ limited: boolean, remaining: number, resetMs: number }}
 */
export function checkRateLimit(ip, pathname, customLimits) {
  startCleanup()
  
  const limit = findLimit(pathname, customLimits)
  if (!limit) {
    return { limited: false, remaining: Infinity, resetMs: 0 }
  }

  const now = Date.now()
  const windowStart = now - limit.windowMs

  if (!requestLog.has(ip)) {
    requestLog.set(ip, new Map())
  }
  
  const ipRoutes = requestLog.get(ip)
  const routeKey = Object.keys(customLimits || DEFAULT_LIMITS).find(
    prefix => pathname.startsWith(prefix)
  ) || pathname

  if (!ipRoutes.has(routeKey)) {
    ipRoutes.set(routeKey, [])
  }

  const timestamps = ipRoutes.get(routeKey)
  
  // Remove timestamps outside window
  const active = timestamps.filter(t => t > windowStart)
  ipRoutes.set(routeKey, active)

  if (active.length >= limit.maxRequests) {
    const oldestInWindow = active[0]
    const resetMs = oldestInWindow + limit.windowMs - now
    return {
      limited: true,
      remaining: 0,
      resetMs: Math.max(0, resetMs),
    }
  }

  // Record this request
  active.push(now)

  return {
    limited: false,
    remaining: limit.maxRequests - active.length,
    resetMs: 0,
  }
}

/**
 * Express/connect-style middleware creator for rate limiting.
 * Returns a function that can be used in the request handler.
 */
export function rateLimitMiddleware(customLimits) {
  return function rateLimit(ip, pathname) {
    return checkRateLimit(ip, pathname, customLimits)
  }
}
