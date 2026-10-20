<template>
  <div>
    <div class="dap-toolbar" style="margin-bottom: 8px">
      <el-button size="small" :icon="Plus" @click="openLibrary">从属性库添加</el-button>
      <el-button size="small" :icon="Plus" @click="newDialog = true">新建属性</el-button>
      <span class="dap-muted">
        属性就是「名称 + 值」：属性库存的是可复用的属性名，值在每次上传时按需填写（值一律按文本存）。
        模板字段之外还能从属性库挑选；属性库里没有的，新建后自动保留、下次直接复用
      </span>
    </div>

    <div v-if="extraItems.length" class="dap-toolbar" style="margin-bottom: 6px">
      <span class="dap-muted">本次关联的属性：</span>
      <el-tag
        v-for="item in extraItems"
        :key="item.name"
        closable
        size="small"
        @close="removeExtra(item.name)"
      >
        {{ item.name }}
      </el-tag>
    </div>

    <AttributeForm
      :items="items"
      :model-value="modelValue"
      @update:model-value="(value) => emit('update:modelValue', value)"
    />

    <el-dialog v-model="libraryDialog" title="从属性库添加属性" width="520px">
      <el-select
        v-model="picked"
        multiple
        filterable
        placeholder="可多选（输入关键字过滤）"
        style="width: 100%"
      >
        <el-option v-for="item in availableDefinitions" :key="item.name" :label="item.name" :value="item.name" />
      </el-select>
      <div class="dap-muted" style="margin-top: 8px">
        列表里没有想要的属性？点「新建属性」，保存后会自动进入属性库并加入本次关联。
      </div>
      <template #footer>
        <el-button @click="libraryDialog = false">取消</el-button>
        <el-button type="primary" :disabled="!picked.length" @click="confirmPicked">添加</el-button>
      </template>
    </el-dialog>

    <AttributeCreateDialog v-model="newDialog" @created="onCreated" />
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import AttributeCreateDialog from './AttributeCreateDialog.vue'
import AttributeForm from './AttributeForm.vue'

const props = defineProps({
  templateItems: { type: Array, default: () => [] },
  definitions: { type: Array, default: () => [] },
  modelValue: { type: Object, default: () => ({}) },
})
const emit = defineEmits(['update:modelValue', 'definitions-changed'])

const extraDefs = ref([])
const libraryDialog = ref(false)
const newDialog = ref(false)
const picked = ref([])

function toItem(definition) {
  return {
    name: definition.name,
    required: false,
  }
}

const extraItems = computed(() => extraDefs.value.map(toItem))

// 模板字段在前，用户从属性库额外关联的在后（按属性名去重）
const items = computed(() => {
  const seen = new Set(props.templateItems.map((item) => item.name))
  return [...props.templateItems, ...extraItems.value.filter((item) => !seen.has(item.name))]
})

const availableDefinitions = computed(() => {
  const used = new Set([
    ...props.templateItems.map((item) => item.name),
    ...extraDefs.value.map((item) => item.name),
  ])
  return props.definitions.filter((item) => !used.has(item.name))
})

function openLibrary() {
  if (!availableDefinitions.value.length) {
    if (props.definitions.length) {
      ElMessage.warning('属性库里的属性都已关联过了')
    } else {
      ElMessage.warning('属性库为空，请先「新建属性」')
      newDialog.value = true
    }
    return
  }
  picked.value = []
  libraryDialog.value = true
}

function confirmPicked() {
  let added = 0
  picked.value.forEach((name) => {
    if (extraDefs.value.some((item) => item.name === name)) return
    const definition = props.definitions.find((item) => item.name === name)
    if (!definition) return
    extraDefs.value.push(definition)
    added += 1
  })
  picked.value = []
  libraryDialog.value = false
  ElMessage.success(`已关联 ${added} 个属性`)
}

function removeExtra(name) {
  extraDefs.value = extraDefs.value.filter((item) => item.name !== name)
  const next = { ...props.modelValue }
  delete next[name]
  emit('update:modelValue', next)
}

function onCreated(definition) {
  const inTemplate = props.templateItems.some((item) => item.name === definition.name)
  if (!inTemplate && !extraDefs.value.some((item) => item.name === definition.name)) {
    extraDefs.value.push(definition)
  }
  emit('definitions-changed')
}
</script>
