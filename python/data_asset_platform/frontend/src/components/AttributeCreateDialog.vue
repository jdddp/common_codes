<template>
  <el-dialog
    :model-value="modelValue"
    title="新建属性（保存后保留在属性库）"
    width="520px"
    @update:model-value="(value) => emit('update:modelValue', value)"
  >
    <el-form label-width="100px">
      <el-form-item label="属性名称" required>
        <el-input
          v-model="form.name"
          placeholder="如 航次、海域"
          @keyup.enter="submit"
        />
      </el-form-item>
    </el-form>
    <div class="dap-muted">
      属性只有名称，值一律按文本保存（不设类型）。<br />
      保存时先查属性库：已有同名属性就直接复用（不会覆盖原有定义），没有才新建并保留，之后上传与检索都能直接复用。
    </div>
    <template #footer>
      <el-button @click="emit('update:modelValue', false)">取消</el-button>
      <el-button type="primary" :loading="creating" @click="submit">保存</el-button>
    </template>
  </el-dialog>
</template>

<script setup>
import { ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api/client'

const props = defineProps({
  modelValue: { type: Boolean, default: false },
})
const emit = defineEmits(['update:modelValue', 'created'])

const creating = ref(false)
const form = ref({ name: '' })

watch(
  () => props.modelValue,
  (value) => {
    if (value) form.value = { name: '' }
  },
)

async function submit() {
  const name = form.value.name.trim()
  if (!name) {
    ElMessage.warning('请填写属性名称')
    return
  }
  if (/[=]/.test(name)) {
    ElMessage.warning('属性名称里不能出现 =（筛选条件里用它分隔属性名和值）')
    return
  }
  creating.value = true
  try {
    const data = await api.resolveAttributes([name])
    const definition = data.items[0]
    const created = data.created.includes(name)
    ElMessage.success(
      created ? `已新建属性「${name}」并保留到属性库` : `属性库已有「${name}」，已直接复用`,
    )
    emit('created', definition)
    emit('update:modelValue', false)
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    creating.value = false
  }
}
</script>
