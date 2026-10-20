<template>
  <div class="dap-page">
    <div class="dap-card" style="margin-bottom: 12px">
      <div class="dap-section-title">检索条件</div>
      <el-form label-width="90px" size="default">
        <el-row :gutter="12">
          <el-col :span="8">
            <el-form-item label="关键词">
              <el-input v-model="form.keyword" placeholder="文件名 / 相对路径" clearable @keyup.enter="reload" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="格式">
              <el-select v-model="form.ext" multiple clearable placeholder="全部" style="width: 100%">
                <el-option v-for="item in extOptions" :key="item" :label="item" :value="item" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="解析状态">
              <el-select v-model="form.parse_status" clearable placeholder="全部" style="width: 100%">
                <el-option label="可在线查看" value="ready" />
                <el-option label="排队中" value="pending" />
                <el-option label="不支持在线查看" value="unsupported" />
                <el-option label="解析失败" value="failed" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="大小(MB)">
              <el-input-number v-model="form.sizeMbMin" :min="0" :controls="false" placeholder="最小" style="width: 45%" />
              <span style="margin: 0 6px">-</span>
              <el-input-number v-model="form.sizeMbMax" :min="0" :controls="false" placeholder="最大" style="width: 45%" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="上传时间">
              <el-date-picker
                v-model="form.dateRange"
                type="daterange"
                value-format="YYYY-MM-DD"
                start-placeholder="开始"
                end-placeholder="结束"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="排序">
              <el-select v-model="form.sort" style="width: 100%">
                <el-option label="上传时间 ↓" value="uploaded_at:desc" />
                <el-option label="上传时间 ↑" value="uploaded_at:asc" />
                <el-option label="大小 ↓" value="size:desc" />
                <el-option label="大小 ↑" value="size:asc" />
                <el-option label="文件名 ↑" value="name:asc" />
                <el-option label="帧数 ↓" value="frame_count:desc" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">自定义属性条件</el-divider>
        <AttrFilterBuilder
          :definitions="definitions"
          :model-value="attrConditions"
          @update:model-value="(value) => (attrConditions = value)"
          @definitions-changed="loadDefinitions"
        />

        <div class="dap-toolbar" style="margin-top: 10px">
          <el-button type="primary" :icon="Search" @click="reload">检索</el-button>
          <el-button :icon="RefreshLeft" @click="resetForm">重置</el-button>
          <el-button :icon="Upload" @click="uploadVisible = true">上传文件</el-button>
          <el-button :icon="FolderOpened" @click="importVisible = true">服务端导入</el-button>
          <div style="flex: 1" />
          <el-button :icon="Download" :disabled="!rows.length" @click="exportCsv">导出清单 CSV</el-button>
        </div>
      </el-form>
    </div>

    <FileTable
      :rows="rows"
      :loading="loading"
      @view="openFile"
      @download="download"
      @parse="parseOne"
      @delete="deleteOne"
      @selection-change="(value) => (selected = value)"
    />

    <div class="dap-toolbar" style="margin-top: 12px">
      <el-pagination
        layout="total, prev, pager, next, sizes"
        :total="total"
        :current-page="page"
        :page-size="pageSize"
        :page-sizes="[20, 50, 100, 200]"
        @current-change="(value) => { page = value; load() }"
        @size-change="(value) => { pageSize = value; page = 1; load() }"
      />
      <div style="flex: 1" />
      <el-button type="danger" :disabled="!selected.length" @click="deleteSelected">
        删除选中（{{ selected.length }}）
      </el-button>
    </div>

    <el-dialog v-model="uploadVisible" title="上传文件" width="880px" top="6vh" destroy-on-close>
      <ChunkUploader v-if="uploadVisible" @done="onUploaded" @close="uploadVisible = false" />
    </el-dialog>

    <el-dialog v-model="importVisible" title="服务端导入（存量数据迁移）" width="760px">
      <el-alert
        type="info"
        :closable="false"
        show-icon
        title="把服务器上已有的文件/目录登记进平台：默认复制到存储（原文件保留），内容相同自动去重"
        style="margin-bottom: 12px"
      />
      <el-form label-width="90px">
        <el-form-item label="路径">
          <el-input
            v-model="importForm.paths"
            type="textarea"
            :rows="4"
            placeholder="服务器上的绝对路径；多个用换行或分号分隔"
          />
        </el-form-item>
        <el-form-item label="根目录名">
          <el-input v-model="importForm.rootName" placeholder="可留空" style="width: 420px" />
        </el-form-item>
        <el-form-item label="选项">
          <el-switch v-model="importForm.recursive" active-text="目录递归" />
          <el-switch v-model="importForm.move" active-text="移动原文件（默认复制，更安全）" style="margin-left: 16px" />
        </el-form-item>
      </el-form>

      <el-table v-if="importResult.length" :data="importResult" size="small" border max-height="240">
        <el-table-column prop="path" label="路径" min-width="240" show-overflow-tooltip />
        <el-table-column label="结果" width="90">
          <template #default="{ row }">
            <el-tag :type="row.status === 'done' ? 'success' : 'danger'" size="small">
              {{ row.status === 'done' ? '成功' : '失败' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="说明" min-width="200">
          <template #default="{ row }">
            <span v-if="row.status === 'done'">
              → {{ row.rel_path }}{{ row.deduped ? '（内容已存在，复用存储）' : '' }}
            </span>
            <span v-else class="dap-muted">{{ row.message }}</span>
          </template>
        </el-table-column>
      </el-table>

      <template #footer>
        <el-button @click="importVisible = false">关闭</el-button>
        <el-button type="primary" :loading="importing" @click="doImport">开始导入</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Download, FolderOpened, RefreshLeft, Search, Upload } from '@element-plus/icons-vue'
import api from '../api/client'
import AttrFilterBuilder from '../components/AttrFilterBuilder.vue'
import ChunkUploader from '../components/ChunkUploader.vue'
import FileTable from '../components/FileTable.vue'

const rows = ref([])
const selected = ref([])
const loading = ref(false)
const total = ref(0)
const page = ref(1)
const pageSize = ref(50)

const definitions = ref([])
const attrConditions = ref([])

const uploadVisible = ref(false)
const importVisible = ref(false)
const importing = ref(false)
const importResult = ref([])
const importForm = ref({ paths: '', rootName: '', move: false, recursive: true })

const extOptions = ['.dat', '.hdy', '.HDY', '.bin', '.tl2', '.xty', '.zip', '.csv', '.txt']

const form = reactive({
  keyword: '',
  ext: [],
  parse_status: '',
  sizeMbMin: null,
  sizeMbMax: null,
  dateRange: null,
  sort: 'uploaded_at:desc',
})

function params() {
  return {
    keyword: form.keyword,
    ext: form.ext,
    parse_status: form.parse_status,
    size_min: form.sizeMbMin ? form.sizeMbMin * 1024 * 1024 : undefined,
    size_max: form.sizeMbMax ? form.sizeMbMax * 1024 * 1024 : undefined,
    uploaded_from: form.dateRange?.[0] ? `${form.dateRange[0]} 00:00:00` : '',
    uploaded_to: form.dateRange?.[1] ? `${form.dateRange[1]} 23:59:59` : '',
    attr: attrConditions.value,
    sort: form.sort,
  }
}

async function load() {
  loading.value = true
  try {
    const data = await api.listFiles({ ...params(), page: page.value, page_size: pageSize.value })
    rows.value = data.items
    total.value = data.total
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loading.value = false
  }
}

function reload() {
  page.value = 1
  load()
}

function resetForm() {
  Object.assign(form, {
    keyword: '',
    ext: [],
    parse_status: '',
    sizeMbMin: null,
    sizeMbMax: null,
    dateRange: null,
    sort: 'uploaded_at:desc',
  })
  attrConditions.value = []
  reload()
}

function openFile(row) {
  window.location.hash = `#/files/${row.id}`
}

function download(row) {
  window.location.href = api.downloadUrl(row.id)
}

function exportCsv() {
  window.location.href = api.exportUrl(params())
}

async function parseOne(row) {
  try {
    const result = await api.parseFile(row.id)
    ElMessage[result.parse_status === 'ready' ? 'success' : 'warning'](
      result.parse_status === 'ready'
        ? `解析成功：${result.frame_count} 帧`
        : result.reason || '该格式暂不支持在线查看',
    )
    await load()
  } catch (error) {
    ElMessage.error(error.message)
  }
}

async function deleteOne(row) {
  try {
    await ElMessageBox.confirm(`确定删除「${row.original_name}」的登记信息吗？`, '删除确认', { type: 'warning' })
    await api.deleteFile(row.id)
    ElMessage.success('已删除')
    await load()
  } catch (error) {
    if (error === 'cancel' || error?.message === 'cancel') return
    ElMessage.error(error.message)
  }
}

async function deleteSelected() {
  try {
    await ElMessageBox.confirm(`确定删除选中的 ${selected.value.length} 个文件吗？`, '批量删除', { type: 'warning' })
    for (const row of selected.value) {
      await api.deleteFile(row.id)
    }
    ElMessage.success('已删除')
    selected.value = []
    await load()
  } catch (error) {
    if (error === 'cancel' || error?.message === 'cancel') return
    ElMessage.error(error.message)
  }
}

async function loadDefinitions() {
  try {
    const data = await api.listAttributes()
    definitions.value = data.items
  } catch (error) {
    /* 忽略 */
  }
}

function onUploaded() {
  ElMessage.success('上传完成')
  uploadVisible.value = false
  reload()
}

async function doImport() {
  const paths = importForm.value.paths
    .split(/[\n;]+/)
    .map((item) => item.trim())
    .filter(Boolean)
  if (!paths.length) {
    ElMessage.warning('请填写服务器上的文件或目录路径')
    return
  }
  importing.value = true
  importResult.value = []
  try {
    const data = await api.importFiles({
      paths,
      root_name: importForm.value.rootName,
      move: importForm.value.move,
      recursive: importForm.value.recursive,
    })
    importResult.value = data.items
    const okCount = data.items.filter((item) => item.status === 'done').length
    ElMessage.success(`导入完成：成功 ${okCount} / ${data.items.length}`)
    await load()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    importing.value = false
  }
}

onMounted(async () => {
  await loadDefinitions()
  await load()
})
</script>
