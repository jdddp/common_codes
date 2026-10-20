<template>
  <div class="dap-page">
    <div class="dap-toolbar">
      <el-page-header :content="file.original_name || '文件详情'" @back="goBack" />
      <div style="flex: 1" />
      <el-button :icon="Download" @click="download">下载原始文件</el-button>
      <el-button :icon="Refresh" :loading="parsing" @click="reparse">重新解析</el-button>
      <el-button :icon="EditPen" @click="renameDialog = true">重命名</el-button>
      <el-button type="danger" :icon="Delete" @click="remove">删除</el-button>
    </div>

    <el-descriptions :column="4" border size="small" style="margin-bottom: 12px">
      <el-descriptions-item label="文件ID">{{ file.id }}</el-descriptions-item>
      <el-descriptions-item label="大小">{{ formatSize(file.size_bytes) }}</el-descriptions-item>
      <el-descriptions-item label="格式">{{ file.format_label || file.ext || '-' }}</el-descriptions-item>
      <el-descriptions-item label="相对路径">{{ file.rel_path || '(根目录)' }}</el-descriptions-item>
      <el-descriptions-item label="上传时间">{{ file.uploaded_at }}</el-descriptions-item>
      <el-descriptions-item label="解析状态">
        <el-tag :type="PARSE_STATUS_TYPE[file.parse_status] || 'info'" size="small">
          {{ parseStatusText(file.parse_status) }}
        </el-tag>
      </el-descriptions-item>
      <el-descriptions-item label="协议">
        {{ PROTOCOL_TEXT[file.protocol] || file.protocol || '-' }}
      </el-descriptions-item>
      <el-descriptions-item label="帧数">{{ file.frame_count ?? '-' }}</el-descriptions-item>
      <el-descriptions-item label="图像尺寸">{{ file.image_shape || '-' }}</el-descriptions-item>
      <el-descriptions-item label="SHA-256" :span="2">
        <span style="font-family: Consolas, monospace; font-size: 12px">{{ file.sha256 }}</span>
      </el-descriptions-item>
      <el-descriptions-item v-if="file.parse_error" label="解析信息" :span="4">
        <span style="color: #d9534f">{{ file.parse_error }}</span>
      </el-descriptions-item>
    </el-descriptions>

    <div class="dap-card" style="margin-bottom: 12px">
      <div class="dap-section-title">
        自定义属性
        <span class="dap-muted" style="font-weight: 400; margin-left: 8px">
          属性只有名称、值一律按文本保存；属性库里没有的名称可点「新建属性」现场建好再填值
        </span>
      </div>
      <div style="margin-bottom: 8px">
        <el-button :icon="Plus" size="small" @click="createDialog = true">新建属性</el-button>
        <span class="dap-muted">新建后保留在属性库，上传与检索都能复用</span>
      </div>
      <AttributeForm
        :items="definitionItems"
        :model-value="attributes"
        @update:model-value="(value) => (attributes = value)"
      />
      <div style="text-align: right">
        <el-button type="primary" :loading="savingAttributes" @click="saveAttributes">保存属性</el-button>
      </div>
    </div>

    <AttributeCreateDialog v-model="createDialog" @created="onAttributeCreated" />

    <div v-if="info.supported" class="dap-card">
      <div class="dap-section-title">
        在线查看
        <span class="dap-muted" style="font-weight: 400; margin-left: 8px">
          后端逐帧解析渲染，前端只负责显示；播放/倍速/算法链与桌面版一致
        </span>
      </div>
      <FrameViewer :file-id="file.id" :info="info" />
    </div>

    <el-alert
      v-else-if="infoLoaded"
      type="warning"
      show-icon
      :closable="false"
      :title="`该文件暂不支持在线查看：${info.reason || '未知格式'}`"
      description="文件本体仍已完整存储在平台上，可直接下载使用；若确需在线查看，请联系管理员扩展对应协议解析器。"
    />

    <el-dialog v-model="renameDialog" title="重命名 / 调整目录" width="520px">
      <el-form label-width="90px">
        <el-form-item label="文件名">
          <el-input v-model="renameForm.original_name" />
        </el-form-item>
        <el-form-item label="相对路径">
          <el-input v-model="renameForm.rel_path" placeholder="例如 raw/nav/day1" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="renameDialog = false">取消</el-button>
        <el-button type="primary" @click="doRename">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, Download, EditPen, Plus, Refresh } from '@element-plus/icons-vue'
import api from '../api/client'
import AttributeCreateDialog from '../components/AttributeCreateDialog.vue'
import AttributeForm from '../components/AttributeForm.vue'
import FrameViewer from '../components/FrameViewer.vue'
import {
  PARSE_STATUS_TYPE,
  PROTOCOL_TEXT,
  formatSize,
  parseStatusText,
} from '../utils/format'

const route = useRoute()
const router = useRouter()

const file = ref({})
const attributes = ref({})
const definitions = ref([])
const info = ref({ supported: false })
const infoLoaded = ref(false)
const parsing = ref(false)
const savingAttributes = ref(false)
const createDialog = ref(false)
const renameDialog = ref(false)
const renameForm = ref({ original_name: '', rel_path: '' })

const definitionItems = computed(() =>
  definitions.value.map((item) => ({
    name: item.name,
    required: false,
  })),
)

async function load() {
  const id = route.params.id
  file.value = await api.getFile(id)
  attributes.value = { ...(file.value.attributes || {}) }
  renameForm.value = { original_name: file.value.original_name, rel_path: file.value.rel_path || '' }
  infoLoaded.value = false
  try {
    info.value = await api.sonarInfo(id)
  } catch (error) {
    info.value = { supported: false, reason: error.message }
  }
  infoLoaded.value = true
}

async function loadDefinitions() {
  try {
    const data = await api.listAttributes()
    definitions.value = data.items
  } catch (error) {
    /* 忽略 */
  }
}

function goBack() {
  if (window.history.length > 1) router.back()
  else router.push('/files')
}

function download() {
  window.location.href = api.downloadUrl(file.value.id)
}

async function reparse() {
  parsing.value = true
  try {
    const result = await api.parseFile(file.value.id)
    if (result.parse_status === 'ready') {
      ElMessage.success(`解析成功：${result.frame_count} 帧，${result.image_shape}`)
    } else {
      ElMessage.warning(result.reason || '该格式暂不支持在线查看')
    }
    await load()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    parsing.value = false
  }
}

async function saveAttributes() {
  savingAttributes.value = true
  try {
    const values = Object.entries(attributes.value).map(([name, value]) => ({ name, value }))
    const result = await api.setFileAttributes(file.value.id, { values, replace: false, create_missing: true })
    file.value = result
    attributes.value = { ...(result.attributes || {}) }
    if (result.ignored_names?.length) {
      ElMessage.warning(`以下属性未定义被忽略：${result.ignored_names.join(', ')}`)
    } else {
      ElMessage.success('属性已保存')
    }
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    savingAttributes.value = false
  }
}

async function doRename() {
  try {
    file.value = await api.renameFile(file.value.id, renameForm.value)
    renameDialog.value = false
    ElMessage.success('已保存')
  } catch (error) {
    ElMessage.error(error.message)
  }
}

async function remove() {
  try {
    await ElMessageBox.confirm('确定删除该文件的登记信息吗？', '删除确认', { type: 'warning' })
    await api.deleteFile(file.value.id)
    ElMessage.success('已删除')
    goBack()
  } catch (error) {
    if (error === 'cancel' || error?.message === 'cancel') return
    ElMessage.error(error.message)
  }
}

watch(() => route.params.id, load)
onMounted(async () => {
  await Promise.all([load(), loadDefinitions()])
})
</script>
