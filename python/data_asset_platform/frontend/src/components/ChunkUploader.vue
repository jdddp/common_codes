<template>
  <div>
    <el-alert
      type="info"
      :closable="false"
      show-icon
      title="支持多文件与整目录上传：分片传输（可断点续传）、按内容去重（同一文件不重复占用空间）、目录层级原样保留。"
      style="margin-bottom: 12px"
    />

    <el-form label-width="110px" style="margin-bottom: 8px">
      <el-form-item label="属性模板">
        <el-select
          v-model="templateId"
          placeholder="不使用模板（可稍后单独设置属性）"
          clearable
          style="width: 320px"
          @change="onTemplateChange"
        >
          <el-option v-for="item in templates" :key="item.id" :label="item.name" :value="item.id" />
        </el-select>
        <span class="dap-muted" style="margin-left: 10px">
          模板决定本批次文件自动加载哪些属性字段（例如深度、海域、设备）
        </span>
      </el-form-item>
      <el-form-item label="根目录名">
        <el-input v-model="rootName" placeholder="可留空；填写后所有文件会归入该目录（避免与已有文件重名）" style="width: 420px" />
      </el-form-item>
    </el-form>

    <AttributePicker
      :template-items="templateItems"
      :definitions="definitions"
      :model-value="attributes"
      @update:model-value="(value) => (attributes = value)"
      @definitions-changed="loadDefinitions"
    />

    <div class="dap-toolbar">
      <input ref="fileInput" type="file" multiple style="display: none" @change="onPickFiles" />
      <input ref="dirInput" type="file" webkitdirectory directory multiple style="display: none" @change="onPickDir" />
      <el-button :icon="Document" @click="fileInput.click()">选择文件</el-button>
      <el-button :icon="FolderOpened" @click="dirInput.click()">选择文件夹</el-button>
      <el-button type="primary" :disabled="!pendingCount || uploading" :loading="uploading" @click="startUpload">
        开始上传（{{ pendingCount }}）
      </el-button>
      <el-button :disabled="!entries.length || uploading" @click="resetAll">清空列表</el-button>
      <span v-if="!entries.length" class="dap-muted">尚未选择文件</span>
      <div style="flex: 1" />
      <span v-if="entries.length" class="dap-muted">
        共 {{ entries.length }} 个文件，合计 {{ formatSize(totalBytes) }}
      </span>
    </div>

    <el-progress
      v-if="entries.length"
      :percentage="overallPercent"
      :status="overallPercent === 100 ? 'success' : undefined"
      style="margin-bottom: 10px"
    />

    <el-table :data="entries" border size="small" max-height="360">
      <el-table-column prop="name" label="文件名" min-width="220" show-overflow-tooltip />
      <el-table-column prop="relPath" label="目录（相对）" min-width="160" show-overflow-tooltip />
      <el-table-column label="大小" width="100" align="right">
        <template #default="{ row }">{{ formatSize(row.size) }}</template>
      </el-table-column>
      <el-table-column label="状态" width="220">
        <template #default="{ row }">
          <el-progress
            v-if="row.status === 'uploading'"
            :percentage="row.percent"
            :stroke-width="10"
            text-inside
          />
          <el-tag v-else-if="row.status === 'done'" type="success" size="small">完成</el-tag>
          <el-tag v-else-if="row.status === 'deduped'" type="success" size="small" effect="plain">秒传</el-tag>
          <el-tag v-else-if="row.status === 'failed'" type="danger" size="small">失败</el-tag>
          <el-tag v-else type="info" size="small" effect="plain">待上传</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="message" label="说明" min-width="180" show-overflow-tooltip />
      <el-table-column label="操作" width="140">
        <template #default="{ row, $index }">
          <el-button size="small" type="primary" link v-if="row.status === 'done'" @click="openFile(row)">
            查看
          </el-button>
          <el-button size="small" type="danger" link :disabled="uploading" @click="removeEntry($index)">
            移除
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <div style="margin-top: 12px; text-align: right">
      <el-button :disabled="uploading" @click="close">关闭</el-button>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Document, FolderOpened } from '@element-plus/icons-vue'
import api from '../api/client'
import AttributePicker from './AttributePicker.vue'
import { formatSize } from '../utils/format'

const emit = defineEmits(['done', 'close'])

const fileInput = ref(null)
const dirInput = ref(null)

const entries = ref([])
const templates = ref([])
const templateId = ref(null)
const templateItems = ref([])
const definitions = ref([])
const attributes = ref({})
const rootName = ref('')
const uploading = ref(false)
const concurrency = 3
const SHA_LIMIT = 64 * 1024 * 1024 // 超过此大小跳过浏览器端哈希（仍由服务端合并时计算）

const pendingCount = computed(() => entries.value.filter((row) => row.status === 'pending').length)
const totalBytes = computed(() => entries.value.reduce((sum, row) => sum + row.size, 0))
const overallPercent = computed(() => {
  if (!entries.value.length) return 0
  const done = entries.value.filter((row) =>
    ['done', 'deduped'].includes(row.status),
  ).length
  const partial = entries.value
    .filter((row) => row.status === 'uploading')
    .reduce((sum, row) => sum + row.percent / 100, 0)
  return Math.floor(((done + partial) / entries.value.length) * 100)
})

onMounted(async () => {
  try {
    const data = await api.listTemplates()
    templates.value = data.items
  } catch (error) {
    /* 忽略 */
  }
  await loadDefinitions()
})

// 属性库：上传时可从中挑选已有属性；新建的属性也会进入这里，供后续复用
async function loadDefinitions() {
  try {
    const data = await api.listAttributes()
    definitions.value = data.items
  } catch (error) {
    /* 忽略 */
  }
}

// 只提交用户真正填过的属性（空值不提交，避免把空字符串写成属性值 / 误建属性）
function collectAttributes() {
  const values = {}
  Object.entries(attributes.value || {}).forEach(([name, value]) => {
    if (value === undefined || value === null || value === '') return
    values[name] = value
  })
  return values
}

async function onTemplateChange(value) {
  attributes.value = {}
  templateItems.value = []
  if (!value) return
  const template = templates.value.find((item) => item.id === value)
  if (!template) return
  templateItems.value = template.items
  template.items.forEach((item) => {
    if (item.default_value !== null && item.default_value !== undefined && item.default_value !== '') {
      attributes.value[item.name] = item.default_value
    }
  })
}

function addEntries(fileList, fromDirectory) {
  const added = []
  Array.from(fileList).forEach((file) => {
    const relative = file.webkitRelativePath || ''
    const parts = relative ? relative.split('/') : []
    const name = file.name
    const relPath = fromDirectory && parts.length > 1 ? parts.slice(0, -1).join('/') : ''
    added.push({
      key: `${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).slice(2, 8)}`,
      file,
      name,
      relPath,
      size: file.size,
      sha256: null,
      status: 'pending',
      percent: 0,
      sessionId: null,
      fileId: null,
      message: '',
    })
  })
  entries.value = fromDirectory ? added : [...entries.value, ...added]
  if (fromDirectory) {
    // 目录上传时层级已包含根目录名，根目录名留空避免重复
    rootName.value = ''
  }
}

function onPickFiles(event) {
  addEntries(event.target.files, false)
  event.target.value = ''
}

function onPickDir(event) {
  addEntries(event.target.files, true)
  event.target.value = ''
}

function removeEntry(index) {
  entries.value.splice(index, 1)
}

function resetAll() {
  if (uploading.value) return
  entries.value = []
  attributes.value = {}
  templateItems.value = []
  templateId.value = null
  rootName.value = ''
}

function close() {
  emit('close')
}

function openFile(row) {
  if (row.fileId) window.open(`#/files/${row.fileId}`, '_self')
}

async function sha256Of(file) {
  if (file.size > SHA_LIMIT) return null
  try {
    if (!window.crypto?.subtle) return null
    const buffer = await file.arrayBuffer()
    const digest = await window.crypto.subtle.digest('SHA-256', buffer)
    return Array.from(new Uint8Array(digest))
      .map((byte) => byte.toString(16).padStart(2, '0'))
      .join('')
  } catch (error) {
    return null
  }
}

async function startUpload() {
  const targets = entries.value.filter((row) => row.status === 'pending')
  if (!targets.length) return
  uploading.value = true
  try {
    // 1) 提前计算小文件哈希，命中内容寻址可直接"秒传"
    for (const row of targets) {
      if (!row.sha256) {
        row.message = '计算校验值…'
        row.sha256 = await sha256Of(row.file)
        row.message = ''
      }
    }

    // 2) 初始化上传会话
    const payload = {
      root_name: rootName.value || '',
      template_id: templateId.value || null,
      attributes: collectAttributes(),
      // 属性库里还没有的属性名由服务端自动补建并保留（前端已先在「新建属性」里登记，这里只是兜底）
      auto_create_attributes: true,
      files: targets.map((row) => ({
        client_key: row.key,
        name: row.name,
        rel_path: row.relPath || '',
        size: row.size,
        sha256: row.sha256,
      })),
    }
    const response = await api.initUpload(payload)

    const sessions = []
    response.items.forEach((item) => {
      const row = targets.find((entry) => entry.key === item.client_key)
      if (!row) return
      if (item.status === 'done') {
        row.status = 'deduped'
        row.fileId = item.file_id
        row.percent = 100
        row.message = item.message || '内容已存在，秒传完成'
        return
      }
      if (item.status === 'failed') {
        row.status = 'failed'
        row.message = item.message || '初始化失败'
        return
      }
      row.status = 'uploading'
      row.sessionId = item.session_id
      row.totalChunks = item.total_chunks
      row.chunkSize = item.chunk_size
      row.received = new Set(item.received || [])
      row.message = item.received?.length ? `续传（已完成 ${item.received.length} 片）` : ''
      sessions.push(row)
    })

    // 3) 并发上传分片
    const queue = [...sessions]
    const workers = Array.from({ length: Math.min(concurrency, queue.length) }, () => worker(queue))
    await Promise.all(workers)

    // 4) 通知已完成的文件
    if (entries.value.some((row) => ['done', 'deduped'].includes(row.status))) {
      emit('done')
    }
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    uploading.value = false
  }
}

async function worker(queue) {
  while (queue.length) {
    const row = queue.shift()
    if (!row) return
    try {
      await uploadOne(row)
    } catch (error) {
      row.status = 'failed'
      row.message = error.message
    }
  }
}

async function uploadOne(row) {
  const total = row.totalChunks
  const chunkSize = row.chunkSize
  for (let index = 0; index < total; index += 1) {
    if (row.received.has(index)) continue
    const start = index * chunkSize
    const blob = row.file.slice(start, Math.min(start + chunkSize, row.size))
    const progress = await api.uploadChunk(row.sessionId, index, blob)
    row.received.add(index)
    row.percent = Math.floor((progress.received.length / total) * 100)
    row.message = `${progress.received.length}/${total} 片`
  }
  const result = await api.completeUpload(row.sessionId)
  row.status = 'done'
  row.fileId = result.file_id
  row.percent = 100
  row.message = result.deduped ? '内容与已有文件相同，已复用存储' : '上传完成，后台正在解析'
}
</script>
