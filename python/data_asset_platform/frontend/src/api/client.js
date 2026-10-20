/**
 * 极简 API 客户端：统一前缀、错误提示、上传分片用裸二进制体。
 * 内网部署无鉴权，因此不带 token（见设计文档 §0 部署假设）。
 */
const BASE = '/api'

async function parseResponse(response) {
  const contentType = response.headers.get('content-type') || ''
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    if (contentType.includes('application/json')) {
      try {
        const body = await response.json()
        detail = body.detail || detail
        if (Array.isArray(detail)) detail = detail.map((d) => d.msg || d).join('; ')
      } catch (error) {
        /* 忽略解析失败 */
      }
    }
    throw new Error(detail)
  }
  if (response.status === 204) return null
  if (contentType.includes('application/json')) return response.json()
  return response.text()
}

function buildQuery(params) {
  const search = new URLSearchParams()
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return
    if (Array.isArray(value)) {
      value.forEach((item) => {
        if (item !== undefined && item !== null && item !== '') search.append(key, item)
      })
    } else {
      search.append(key, value)
    }
  })
  const text = search.toString()
  return text ? `?${text}` : ''
}

async function request(method, path, { params, body, raw } = {}) {
  const options = { method, headers: {} }
  if (body !== undefined) {
    if (raw) {
      options.body = body
      options.headers['Content-Type'] = 'application/octet-stream'
    } else {
      options.headers['Content-Type'] = 'application/json'
      options.body = JSON.stringify(body)
    }
  }
  const response = await fetch(`${BASE}${path}${buildQuery(params)}`, options)
  return parseResponse(response)
}

export const api = {
  get: (path, params) => request('GET', path, { params }),
  post: (path, body, params) => request('POST', path, { body, params }),
  put: (path, body, params) => request('PUT', path, { body, params }),
  patch: (path, body, params) => request('PATCH', path, { body, params }),
  del: (path, params) => request('DELETE', path, { params }),

  // ---------------- 属性 / 模板 ----------------
  listAttributes: () => request('GET', '/attributes'),
  createAttribute: (body) => request('POST', '/attributes', { body }),
  // 批量"取或建"：属性库已有则复用，没有才新建（新建后保留，供以后复用）。传属性名数组
  resolveAttributes: (names) => request('POST', '/attributes/resolve', { body: { names } }),
  updateAttribute: (id, body) => request('PATCH', `/attributes/${id}`, { body }),
  deleteAttribute: (id) => request('DELETE', `/attributes/${id}`),
  // 属性"已有的取值"（去重 + 计数）：填写/筛选时直接选，不用手敲
  attributeValues: (params) => request('GET', '/attributes/values', { params }),
  listTemplates: () => request('GET', '/templates'),
  createTemplate: (body) => request('POST', '/templates', { body }),
  updateTemplate: (id, body) => request('PATCH', `/templates/${id}`, { body }),
  deleteTemplate: (id) => request('DELETE', `/templates/${id}`),

  // ---------------- 上传 ----------------
  initUpload: (body) => request('POST', '/uploads/init', { body }),
  uploadChunk: (sessionId, index, blob) =>
    request('PUT', `/uploads/${sessionId}/chunks/${index}`, { body: blob, raw: true }),
  completeUpload: (sessionId) => request('POST', `/uploads/${sessionId}/complete`),
  resetUpload: (sessionId) => request('POST', `/uploads/${sessionId}/reset`),
  abortUpload: (sessionId) => request('DELETE', `/uploads/${sessionId}`),
  uploadProgress: (sessionId) => request('GET', `/uploads/${sessionId}`),

  // ---------------- 服务端导入（存量数据迁移） ----------------
  importFiles: (body) => request('POST', '/import', { body }),

  // ---------------- 文件 ----------------
  listFiles: (params) => request('GET', '/files', { params }),
  getFile: (id) => request('GET', `/files/${id}`),
  renameFile: (id, body) => request('PATCH', `/files/${id}`, { body }),
  setFileAttributes: (id, body) => request('PUT', `/files/${id}/attributes`, { body }),
  parseFile: (id) => request('POST', `/files/${id}/parse`),
  deleteFile: (id) => request('DELETE', `/files/${id}`),
  exportUrl: (params) => `${BASE}/files/export.csv${buildQuery(params)}`,
  downloadUrl: (id) => `${BASE}/files/${id}/download`,
  thumbnailUrl: (id) => `${BASE}/files/${id}/thumbnail`,

  // ---------------- 在线查看 ----------------
  sonarAlgorithms: () => request('GET', '/sonar/algorithms'),
  sonarInfo: (id) => request('GET', `/sonar/${id}/info`),
  sonarView: (id, params) => request('GET', `/sonar/${id}/view`, { params }),
  sonarFrames: (id, params) => request('GET', `/sonar/${id}/frames`, { params }),
  sonarPixel: (id, params) => request('GET', `/sonar/${id}/pixel`, { params }),
  sonarStats: (id) => request('GET', `/sonar/${id}/stats`),
  sonarFrameUrl: (id, index, params) =>
    `${BASE}/sonar/${id}/frame/${index}.png${buildQuery(params)}`,
}

export default api
