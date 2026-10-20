export function formatSize(bytes) {
  if (bytes === null || bytes === undefined) return '-'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = Number(bytes)
  let index = 0
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024
    index += 1
  }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

export function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return Number(value).toFixed(digits)
}

export function formatTime(value) {
  if (!value) return '-'
  if (typeof value === 'number') {
    if (value > 1e12) return new Date(value).toLocaleString('zh-CN')
    if (value > 1e9) return new Date(value * 1000).toLocaleString('zh-CN')
    return String(value)
  }
  return String(value)
}

export const PARSE_STATUS_TEXT = {
  none: '未解析',
  pending: '排队中',
  ready: '可在线查看',
  unsupported: '不支持在线查看',
  failed: '解析失败',
}

export function parseStatusText(status) {
  return PARSE_STATUS_TEXT[status] || status || '-'
}

export const PARSE_STATUS_TYPE = {
  ready: 'success',
  pending: 'warning',
  unsupported: 'info',
  failed: 'danger',
  none: 'info',
}

export const PROTOCOL_TEXT = {
  hdy: 'HDY',
  fishing: 'Fishing',
  norbit: 'Norbit',
  tl2: 'TL2',
  xty: 'XTY',
}
