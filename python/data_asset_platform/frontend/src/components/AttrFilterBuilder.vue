<template>
  <div>
    <div v-for="(row, index) in rows" :key="index" class="dap-toolbar" style="margin-bottom: 6px">
      <el-select v-model="row.name" placeholder="属性" filterable style="width: 200px" @change="emitChange">
        <el-option
          v-for="item in definitions"
          :key="item.name"
          :label="item.name"
          :value="item.name"
        />
      </el-select>
      <span class="dap-muted" style="margin: 0 8px">等于</span>
      <el-select
        v-model="row.value"
        clearable
        filterable
        allow-create
        default-first-option
        :placeholder="placeholderOf(row)"
        style="width: 260px"
        @change="emitChange"
        @visible-change="(visible) => visible && loadValueOptions()"
      >
        <el-option
          v-for="option in valueChoices(row.name)"
          :key="option.value"
          :label="option.label"
          :value="option.value"
        />
      </el-select>
      <el-button :icon="Delete" circle size="small" @click="removeRow(index)" />
    </div>

    <div class="dap-toolbar">
      <el-button size="small" :icon="Plus" @click="addRow">添加属性条件</el-button>
      <el-button size="small" :icon="Plus" @click="openLibrary">从属性库批量添加</el-button>
      <el-button size="small" :icon="Plus" @click="newDialog = true">新建属性</el-button>
      <el-button v-if="rows.length" size="small" link @click="clearRows">清空</el-button>
      <span v-if="!definitions.length" class="dap-muted">
        尚未定义任何属性：可点「新建属性」创建，保存后会保留在属性库
      </span>
    </div>

    <el-dialog v-model="libraryDialog" title="从属性库批量添加条件" width="560px">
      <el-select
        v-model="picked"
        multiple
        filterable
        placeholder="可多选（输入关键字过滤）"
        style="width: 100%"
      >
        <el-option v-for="item in definitions" :key="item.name" :label="item.name" :value="item.name" />
      </el-select>
      <div class="dap-muted" style="margin-top: 8px">
        勾选后每个属性生成一条条件（值可稍后填写）；属性库里没有的，点「新建属性」，保存后保留在属性库。
      </div>
      <template #footer>
        <el-button @click="libraryDialog = false">取消</el-button>
        <el-button type="primary" :disabled="!picked.length" @click="confirmPicked">添加条件</el-button>
      </template>
    </el-dialog>

    <AttributeCreateDialog v-model="newDialog" @created="onCreated" />
  </div>
</template>

<script setup>
import { onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Delete, Plus } from '@element-plus/icons-vue'
import api from '../api/client'
import AttributeCreateDialog from './AttributeCreateDialog.vue'

const props = defineProps({
  definitions: { type: Array, default: () => [] },
  modelValue: { type: Array, default: () => [] },
})
const emit = defineEmits(['update:modelValue', 'definitions-changed'])

const rows = ref([])
const libraryDialog = ref(false)
const newDialog = ref(false)
const picked = ref([])
const valueOptions = ref({})   // 属性名 -> 库里已出现过的取值 [{value, file_count}]

// 值的候选：属性库里"已经填过的值"（按出现次数），供直接选而不是手敲
async function loadValueOptions() {
  try {
    const data = await api.attributeValues({ limit: 200 })
    const map = {}
    ;(data.items || []).forEach((item) => {
      map[item.name] = item.items || []
    })
    valueOptions.value = map
  } catch (error) {
    /* 取不到建议也不影响手输 */
  }
}

onMounted(loadValueOptions)
watch(() => props.definitions, loadValueOptions)
defineExpose({ refreshValueOptions: loadValueOptions })

// 值都是字符串：只有「等于」这一种比较
function placeholderOf() {
  return '值（可从已有值里挑，也可手输）'
}

// 已有取值（带出现次数），并去重
function valueChoices(name) {
  const seen = new Map()
  ;(valueOptions.value[name] || []).forEach((entry) => {
    const text = String(entry.value)
    const count = Number(entry.file_count || 0)
    seen.set(text, { value: text, label: count > 1 ? `${text}（${count} 条）` : text })
  })
  return Array.from(seen.values())
}

function addRow() {
  rows.value.push({
    name: props.definitions[0]?.name || '',
    value: '',
  })
  emitChange()
}

function removeRow(index) {
  rows.value.splice(index, 1)
  emitChange()
}

function clearRows() {
  rows.value = []
  emitChange()
}

function openLibrary() {
  if (!props.definitions.length) {
    ElMessage.warning('属性库为空，请先「新建属性」')
    newDialog.value = true
    return
  }
  picked.value = []
  libraryDialog.value = true
}

// 一次勾选多个属性 → 批量生成条件行（"筛选时关联多个属性"）
function confirmPicked() {
  let added = 0
  picked.value.forEach((name) => {
    if (rows.value.some((row) => row.name === name)) return
    rows.value.push({ name, value: '' })
    added += 1
  })
  emitChange()
  picked.value = []
  libraryDialog.value = false
  ElMessage.success(`已添加 ${added} 个属性条件`)
}

// 新建的属性即时可用于筛选
function onCreated(definition) {
  emit('definitions-changed')
  if (!rows.value.some((row) => row.name === definition.name)) {
    rows.value.push({ name: definition.name, value: '' })
    emitChange()
  }
}

function emitChange() {
  const conditions = []
  rows.value.forEach((row) => {
    if (!row.name || row.value === undefined || row.value === null || row.value === '') return
    conditions.push(`${row.name}=${row.value}`)   // 只有「等于」
  })
  emit('update:modelValue', conditions)
}

// 外部重置（例如清空全部条件）时同步内部行
watch(
  () => props.modelValue,
  (value) => {
    if (!value || !value.length) {
      if (rows.value.length) rows.value = []
    }
  },
)
</script>
