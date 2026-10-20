<template>
  <div class="dap-page">
    <div class="dap-card">
      <el-tabs v-model="tab">
        <!-- ============================ 属性定义 ============================ -->
        <el-tab-pane label="属性定义" name="attributes">
          <div class="dap-toolbar">
            <el-alert
              type="info"
              :closable="false"
              show-icon
              title="属性只有名称（值一律按文本保存，不设类型）；名称即标识，同名即同一个属性。这里新增的属性会出现在属性模板、上传表单、文件详情与检索条件中。"
              style="flex: 1"
            />
            <el-button type="primary" :icon="Plus" @click="openAttributeDialog()">新增属性</el-button>
          </div>

          <el-table :data="definitions" v-loading="loading.attributes" border stripe size="small">
            <el-table-column prop="id" label="ID" width="70" />
            <el-table-column prop="name" label="属性名" min-width="200" />
            <el-table-column prop="used_count" label="已用" width="90" align="right" />
            <el-table-column prop="created_at" label="创建时间" width="170" />
            <el-table-column label="操作" width="150" fixed="right">
              <template #default="{ row }">
                <el-button size="small" link type="primary" @click="openAttributeDialog(row)">编辑</el-button>
                <el-button size="small" link type="danger" @click="removeAttribute(row)">删除</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>

        <!-- ============================ 属性模板 ============================ -->
        <el-tab-pane label="属性模板" name="templates">
          <div class="dap-toolbar">
            <el-alert
              type="info"
              :closable="false"
              show-icon
              title="模板按业务场景组织属性：上传时可一键加载，保证同类数据属性结构一致（可设必填与默认值）。"
              style="flex: 1"
            />
            <el-button type="primary" :icon="Plus" @click="openTemplateDialog()">新建模板</el-button>
          </div>

          <el-table :data="templates" v-loading="loading.templates" border stripe size="small">
            <el-table-column prop="id" label="ID" width="70" />
            <el-table-column prop="name" label="模板名" width="200" />
            <el-table-column prop="description" label="说明" min-width="200" show-overflow-tooltip />
            <el-table-column label="字段" min-width="260">
              <template #default="{ row }">
                <el-tag
                  v-for="item in row.items"
                  :key="item.name"
                  size="small"
                  :type="item.required ? 'warning' : 'info'"
                  effect="plain"
                  style="margin: 0 4px 2px 0"
                >
                  {{ item.name }}<span v-if="item.required"> *</span>
                </el-tag>
                <span v-if="!row.items.length" class="dap-muted">未配置字段</span>
              </template>
            </el-table-column>
            <el-table-column prop="created_at" label="创建时间" width="160" />
            <el-table-column label="操作" width="150" fixed="right">
              <template #default="{ row }">
                <el-button size="small" link type="primary" @click="openTemplateDialog(row)">编辑</el-button>
                <el-button size="small" link type="danger" @click="removeTemplate(row)">删除</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>
      </el-tabs>
    </div>

    <!-- ============================ 属性编辑弹窗 ============================ -->
    <el-dialog v-model="attributeDialog" :title="attributeForm.id ? '编辑属性' : '新增属性'" width="520px">
      <el-form label-width="100px">
        <el-form-item label="属性名" required>
          <el-input v-model="attributeForm.name" placeholder="如 深度、海域" />
        </el-form-item>
      </el-form>
      <div class="dap-muted">
        属性只有名称，值一律按文本保存（不设类型、单位、可选值）。名称里不能出现 =（筛选条件里用它分隔属性名和值）。
      </div>
      <template #footer>
        <el-button @click="attributeDialog = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="saveAttribute">保存</el-button>
      </template>
    </el-dialog>

    <!-- ============================ 模板编辑弹窗 ============================ -->
    <el-dialog v-model="templateDialog" :title="templateForm.id ? '编辑模板' : '新建模板'" width="760px">
      <el-form label-width="100px">
        <el-form-item label="模板名" required>
          <el-input v-model="templateForm.name" placeholder="如 声呐数据模板" />
        </el-form-item>
        <el-form-item label="说明">
          <el-input v-model="templateForm.description" type="textarea" :rows="2" />
        </el-form-item>
        <el-form-item label="字段">
          <div style="width: 100%">
            <div v-for="(item, index) in templateForm.items" :key="index" class="dap-toolbar" style="margin-bottom: 6px">
              <el-select v-model="item.attr_def_id" filterable placeholder="选择属性" style="width: 240px">
                <el-option
                  v-for="definition in definitions"
                  :key="definition.id"
                  :label="definition.name"
                  :value="definition.id"
                />
              </el-select>
              <el-switch v-model="item.required" active-text="必填" />
              <el-input v-model="item.default_value" placeholder="默认值（可空）" style="width: 200px" />
              <el-input-number v-model="item.sort_order" :min="0" size="small" style="width: 110px" />
              <el-button :icon="Delete" circle size="small" @click="templateForm.items.splice(index, 1)" />
            </div>
            <el-button size="small" :icon="Plus" @click="addTemplateItem">添加字段</el-button>
          </div>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="templateDialog = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="saveTemplate">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, Plus } from '@element-plus/icons-vue'
import api from '../api/client'

const tab = ref('attributes')
const definitions = ref([])
const templates = ref([])
const loading = reactive({ attributes: false, templates: false })
const saving = ref(false)

const attributeDialog = ref(false)
const attributeForm = ref({ id: null, name: '' })

const templateDialog = ref(false)
const templateForm = ref({ id: null, name: '', description: '', items: [] })

async function loadAttributes() {
  loading.attributes = true
  try {
    const data = await api.listAttributes()
    definitions.value = data.items
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loading.attributes = false
  }
}

async function loadTemplates() {
  loading.templates = true
  try {
    const data = await api.listTemplates()
    templates.value = data.items
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loading.templates = false
  }
}

function openAttributeDialog(row) {
  attributeForm.value = row ? { id: row.id, name: row.name } : { id: null, name: '' }
  attributeDialog.value = true
}

async function saveAttribute() {
  const form = attributeForm.value
  const name = form.name.trim()
  if (!name) {
    ElMessage.warning('请填写属性名')
    return
  }
  if (/[=]/.test(name)) {
    ElMessage.warning('属性名里不能出现 =（筛选条件里用它分隔属性名和值）')
    return
  }
  saving.value = true
  try {
    const payload = { name }
    if (form.id) await api.updateAttribute(form.id, payload)
    else await api.createAttribute(payload)
    ElMessage.success('已保存')
    attributeDialog.value = false
    await loadAttributes()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    saving.value = false
  }
}

async function removeAttribute(row) {
  try {
    await ElMessageBox.confirm(
      `确定从属性库删除「${row.name}」吗？删除后，上传与检索里都不再出现。已经有关联取值的属性不允许删除（需先清空文件上的该属性值）。`,
      '删除确认',
      { type: 'warning' },
    )
    await api.deleteAttribute(row.id)
    ElMessage.success('已删除')
    await Promise.all([loadAttributes(), loadTemplates()])
  } catch (error) {
    if (error === 'cancel' || error?.message === 'cancel') return
    ElMessage.error(error.message)
  }
}

function openTemplateDialog(row) {
  if (row) {
    templateForm.value = {
      id: row.id,
      name: row.name,
      description: row.description,
      items: row.items.map((item) => ({
        attr_def_id: item.attr_def_id,
        required: item.required,
        default_value: item.default_value ?? '',
        sort_order: item.sort_order,
      })),
    }
  } else {
    templateForm.value = { id: null, name: '', description: '', items: [] }
  }
  templateDialog.value = true
}

function addTemplateItem() {
  templateForm.value.items.push({
    attr_def_id: definitions.value[0]?.id ?? null,
    required: false,
    default_value: '',
    sort_order: templateForm.value.items.length,
  })
}

async function saveTemplate() {
  const form = templateForm.value
  if (!form.name.trim()) {
    ElMessage.warning('请填写模板名')
    return
  }
  if (!form.items.length) {
    ElMessage.warning('请至少添加一个字段')
    return
  }
  saving.value = true
  try {
    const payload = {
      name: form.name.trim(),
      description: form.description,
      items: form.items.map((item, index) => ({
        attr_def_id: item.attr_def_id,
        required: Boolean(item.required),
        default_value: item.default_value === '' ? null : String(item.default_value),
        sort_order: item.sort_order ?? index,
      })),
    }
    if (form.id) await api.updateTemplate(form.id, payload)
    else await api.createTemplate(payload)
    ElMessage.success('已保存')
    templateDialog.value = false
    await loadTemplates()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    saving.value = false
  }
}

async function removeTemplate(row) {
  try {
    await ElMessageBox.confirm(`确定删除模板「${row.name}」吗？`, '删除确认', { type: 'warning' })
    await api.deleteTemplate(row.id)
    ElMessage.success('已删除')
    await loadTemplates()
  } catch (error) {
    if (error === 'cancel' || error?.message === 'cancel') return
    ElMessage.error(error.message)
  }
}

onMounted(async () => {
  await Promise.all([loadAttributes(), loadTemplates()])
})
</script>
