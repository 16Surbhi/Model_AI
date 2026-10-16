import { readFileSync, existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(__dirname, '..', '..')
const envPath = path.join(projectRoot, '.env')

function stripInlineComment(value) {
  let quote = null

  for (let i = 0; i < value.length; i++) {
    const char = value[i]
    if ((char === '"' || char === "'") && value[i - 1] !== '\\') {
      quote = quote === char ? null : quote || char
    }
    if (char === '#' && !quote && /\s/.test(value[i - 1] || ' ')) {
      return value.slice(0, i).trim()
    }
  }

  return value.trim()
}

function cleanEnvValue(value) {
  const stripped = stripInlineComment(value)
  const quote = stripped[0]

  if ((quote === '"' || quote === "'") && stripped[stripped.length - 1] === quote) {
    return stripped.slice(1, -1)
  }

  return stripped
}

export function loadEnvFile(filePath = envPath) {
  if (!existsSync(filePath)) return false

  const raw = readFileSync(filePath, 'utf8')

  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue

    const equalsIndex = trimmed.indexOf('=')
    if (equalsIndex === -1) continue

    const key = trimmed.slice(0, equalsIndex).trim()
    const value = cleanEnvValue(trimmed.slice(equalsIndex + 1))

    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) continue
    if (process.env[key] === undefined) {
      process.env[key] = value
    }
  }

  return true
}

loadEnvFile()
