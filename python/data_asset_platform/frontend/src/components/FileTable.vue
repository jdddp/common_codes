<template>
  <el-table
    :data="rows"
    v-loading="loading"
    border
    stripe
    size="small"
    @selection-change="(value) => emit('selection-change', value)"
  >
    <el-table-column v-if="selectable" type="selection" width="42" />
    <el-table-column label="预览" width="96">
      <template #default="{ row }">
        <el-image
          v-if="row.parse_status === 'ready'"
          :src="api.thumbnailUrl(row.id)"
          fit="contain"
          style="width: 72px; height: 40px; background: #111"
          :preview-src-list="[api.thumbnailUrl(row.id)]"
          preview-teleported
        />
        <span v-else class="dap-muted">无</span>
      </template>
    </el-table-column>
    <el-table-column prop="original_name" label="文件名" min-width="200" show-overflow-tooltip>
      <template #default="{ row }">
        <el-link type="primary" @click="emit('view', row)">{{ row.original_name }}</el-link>
      </template>
    </el-table-column>
    <el-table-column prop="rel_path" label="目录" min-width="130" show-overflow-tooltip />
    <el-table-column label="属性" min-width="200">
      <template #default="{ row }">
        <el-tag
          v-for="(value, name) in row.attributes"
          :key="name"
          size="small"
          effect="plain"
          style="margin: 0 4px 2px 0"
        >
          {{ name }}: {{ value }}
        </el-tag>
        <span v-if="!Object.keys(row.attributes || {}).length" class="dap-muted">-</span>
      </template>
    </el-table-column>
    <el-table-column label="大小" width="98" align="right">
      <template #default="{ row }">{{ formatSize(row.size_bytes) }}</template>
    </el-table-column>
    <el-table-column label="协议" width="86">
      <template #default="{ row }">
        <span v-if="row.protocol">{{ PROTOCOL_TEXT[row.protocol] || row.protocol }}</span>
        <span v-else class="dap-muted">-</span>
      </template>
    </el-table-column>
    <el-table-column label="帧数" width="80" align="right">
      <template #default="{ row }">{{ row.frame_count ?? '-' }}</template>
    </el-table-column>
    <el-table-column label="状态" width="130">
      <template #default="{ row }">
        <el-tooltip :content="row.parse_error || parseStatusText(row.parse_status)" placement="top">
          <el-tag :type="PARSE_STATUS_TYPE[row.parse_status] || 'info'" size="small" effect="light">
            {{ parseStatusText(row.parse_status) }}
          </el-tag>
        </el-tooltip>
      </template>
    </el-table-column>
    <el-table-column prop="uploaded_at" label="上传时间" width="150" />
    <el-table-column label="操作" width="230" fixed="right">
      <template #default="{ row }">
        <el-button size="small" link type="primary" @click="emit('view', row)">
          {{ row.parse_status === 'ready' ? '查看' : '详情' }}
        </el-button>
        <el-button size="small" link type="primary" @click="emit('download', row)">下载</el-button>
        <el-button size="small" link @click="emit('parse', row)">重解析</el-button>
        <el-button size="small" link type="danger" @click="emit('delete', row)">删除</el-button>
      </template>
    </el-table-column>
  </el-table>
</template>

<script setup>
import api from '../api/client'
import { PARSE_STATUS_TYPE, PROTOCOL_TEXT, formatSize, parseStatusText } from '../utils/format'

defineProps({
  rows: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
  selectable: { type: Boolean, default: true },
})
const emit = defineEmits(['view', 'download', 'parse', 'delete', 'selection-change'])
</script>
