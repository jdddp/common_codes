<template>
  <el-form label-width="130px" style="margin-bottom: 4px">
    <el-form-item v-for="item in items" :key="item.name" :required="item.required">
      <template #label>
        <span>{{ item.name }}</span>
      </template>

      <!-- 属性只有名称；值一律是文本：可直接从"已有的取值"里选，也可输入新值 -->
      <el-select
        :model-value="stringValue(item.name)"
        clearable
        filterable
        allow-create
        default-first-option
        :placeholder="choicesOf(item.name).length ? '可直接从已有取值里选，也可输入新值' : '暂无已有取值，直接填写即可'"
        style="width: 420px"
        @update:model-value="(value) => setValue(item.name, value)"
        @visible-change="(visible) => visible && loadValueOptions()"
      >
        <el-option
          v-for="option in choicesOf(item.name)"
          :key="option.value"
          :label="option.label"
          :value="option.value"
        />
      </el-select>
      <span class="dap-muted" style="margin-left: 8px">值按文本保存</span>
    </el-form-item>
    <el-empty v-if="!items.length" description="尚未关联属性（可选）：可从属性库添加，或新建一个属性" :image-size="60" />
  </el-form>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import api from '../api/client'

const props = defineProps({
  items: { type: Array, default: () => [] },
  modelValue: { type: Object, default: () => ({}) },
})
const emit = defineEmits(['update:modelValue'])

const valueOptions = ref({})   // 属性名 -> 库里已出现过的取值 [{value, file_count}]

// 已填过的值直接可选（仍然允许输入新值）
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

function choicesOf(name) {
  const seen = new Map()
  ;(valueOptions.value[name] || []).forEach((entry) => {
    const text = String(entry.value)
    const count = Number(entry.file_count || 0)
    seen.set(text, { value: text, label: count > 1 ? `${text}（${count} 条）` : text })
  })
  return Array.from(seen.values())
}

onMounted(loadValueOptions)
defineExpose({ refreshValueOptions: loadValueOptions })

function setValue(name, value) {
  emit('update:modelValue', { ...props.modelValue, [name]: value })
}

function stringValue(name) {
  const value = props.modelValue[name]
  if (value === undefined || value === null) return ''
  return typeof value === 'object' ? JSON.stringify(value) : String(value)
}
</script>
