<template>
  <div class="viewer-shell">
    <!-- ------------------------------ 播放控制 ------------------------------ -->
    <div class="viewer-toolbar">
      <el-button-group>
        <el-button :icon="DArrowLeft" @click="stop">停止</el-button>
        <el-button :icon="ArrowLeft" @click="stepFrame(-1)">上一帧</el-button>
        <el-button v-if="!playing" type="primary" :icon="VideoPlay" @click="play">播放</el-button>
        <el-button v-else type="warning" :icon="VideoPause" @click="pause">暂停</el-button>
        <el-button :icon="ArrowRight" @click="stepFrame(1)">下一帧</el-button>
      </el-button-group>

      <span style="display: flex; align-items: center; gap: 6px">
        <span class="dap-muted">帧</span>
        <el-input-number
          v-model="jumpIndex"
          :min="1"
          :max="frameCount"
          :controls="false"
          size="small"
          style="width: 88px"
          @change="onJump"
        />
        <span class="dap-muted">/ {{ frameCount }}</span>
      </span>

      <span style="display: flex; align-items: center; gap: 6px">
        <span class="dap-muted">倍速</span>
        <el-select v-model="speed" size="small" style="width: 92px">
          <el-option v-for="item in speedOptions" :key="item" :label="`${item}x`" :value="item" />
        </el-select>
        <span class="dap-muted">{{ intervalMs }}ms/帧 · 步长 {{ stepSize }}</span>
      </span>

      <el-slider
        v-model="jumpIndex"
        :min="1"
        :max="Math.max(frameCount, 1)"
        :show-tooltip="false"
        style="flex: 1; min-width: 160px; margin: 0 12px"
        @input="pause"
        @change="onJump"
      />

      <el-button :icon="RefreshRight" @click="resetDisplay">重置显示</el-button>
    </div>

    <!-- ------------------------------ 显示设置 ------------------------------ -->
    <div class="viewer-toolbar" style="margin-top: 8px">
      <span style="display: flex; align-items: center; gap: 6px">
        <span class="dap-muted">伪彩</span>
        <el-select v-model="colormap" size="small" style="width: 128px" @change="onParamsChanged">
          <el-option v-for="item in colormaps" :key="item" :label="item" :value="item" />
        </el-select>
      </span>

      <span style="display: flex; align-items: center; gap: 6px">
        <span class="dap-muted">亮度</span>
        <el-slider
          v-model="brightness"
          :min="-100"
          :max="100"
          style="width: 150px"
          @input="onAdjustInput"
          @change="onParamsChanged"
        />
        <span class="viewer-readout" style="min-width: 34px">{{ brightness }}</span>
      </span>

      <span style="display: flex; align-items: center; gap: 6px">
        <span class="dap-muted">对比度</span>
        <el-slider
          v-model="contrast"
          :min="10"
          :max="300"
          style="width: 150px"
          @input="onAdjustInput"
          @change="onParamsChanged"
        />
        <span class="viewer-readout" style="min-width: 40px">{{ (contrast / 100).toFixed(2) }}</span>
      </span>

      <el-tooltip content="把极坐标声呐数据映射为扇形显示（需该协议提供波束角）" placement="top">
        <span style="display: flex; align-items: center; gap: 6px">
          <el-switch v-model="sectorFan" :disabled="!supportsSectorFan" @change="onParamsChanged" />
          <span class="dap-muted">扇形映射</span>
        </span>
      </el-tooltip>

      <el-tag size="small" type="info" effect="dark">{{ protocolLabel }}</el-tag>
      <el-tag v-if="loading" size="small" type="warning" effect="dark">取帧中…</el-tag>
      <div style="flex: 1" />
      <span class="dap-muted">{{ statusText }}</span>
    </div>

    <div style="display: flex; gap: 12px; margin-top: 12px; align-items: flex-start">
      <!-- ------------------------------ 算法面板 ------------------------------ -->
      <div class="viewer-panel" style="flex: 0 0 240px">
        <div class="viewer-panel-title">
          <span>算法链（逐帧、无状态）</span>
          <el-button size="small" link type="primary" @click="clearAlgorithms">清空</el-button>
        </div>
        <el-checkbox-group v-model="enabledAlgos" @change="onParamsChanged">
          <div class="viewer-algo-list">
            <el-tooltip
              v-for="item in algorithms"
              :key="item.name"
              :content="item.description"
              placement="right"
              :show-after="300"
            >
              <el-checkbox :label="item.name" :value="item.name">
                {{ item.name }}
              </el-checkbox>
            </el-tooltip>
          </div>
        </el-checkbox-group>
        <div class="dap-muted" style="margin-top: 8px; font-size: 11px">
          顺序即执行顺序：截断 → 亮度 → 对比 → Gamma → CLAHE → 背景校正 → 扇形映射
        </div>
      </div>

      <!-- ------------------------------ 双栏显示 ------------------------------ -->
      <div style="flex: 1; min-width: 0">
        <div class="viewer-panels">
          <div class="viewer-panel">
            <div class="viewer-panel-title">
              <span>灰度（原始数据，不跑算法）</span>
              <span>{{ shapeText }}</span>
            </div>
            <div class="viewer-canvas" :style="canvasStyle">
              <img
                v-if="current && current.gray"
                ref="grayImage"
                :src="`data:${current.gray.mime};base64,${current.gray.data}`"
                alt="gray"
                @mousemove="onHover($event, 'gray')"
                @mouseleave="clearHover"
              />
              <div
                v-if="hoverPixel && hoverPanel === 'gray'"
                class="viewer-crosshair"
                :style="crosshairStyle"
              />
            </div>
            <div class="viewer-readout">
              <template v-if="hoverPixel">
                x={{ hoverPixel.x }} y={{ hoverPixel.y }} · 原始值={{ hoverPixel.raw }}
                <span v-if="hoverPixel.angle_deg !== null">
                  · 波束角={{ hoverPixel.angle_deg.toFixed ? hoverPixel.angle_deg.toFixed(2) : hoverPixel.angle_deg }}°
                </span>
              </template>
              <template v-else>将鼠标移到图像上查看原始像素值</template>
            </div>
          </div>

          <div class="viewer-panel">
            <div class="viewer-panel-title">
              <span>伪彩（算法链 + {{ colormap }}）</span>
              <span>{{ colorShapeText }} · ping {{ current?.meta?.ping_num ?? '-' }}</span>
            </div>
            <div class="viewer-canvas" :style="canvasStyle">
              <img
                v-if="current && current.color"
                :src="`data:${current.color.mime};base64,${current.color.data}`"
                alt="color"
                @mousemove="onHover($event, 'color')"
                @mouseleave="clearHover"
              />
              <div
                v-if="hoverPixel && hoverPanel === 'color'"
                class="viewer-crosshair"
                :style="crosshairStyle"
              />
            </div>
            <div class="viewer-readout">
              <template v-if="hoverPixel">
                x={{ hoverPixel.x }} y={{ hoverPixel.y }} · 处理值={{ hoverPixel.processed === null ? '—' : hoverPixel.processed }}
                <span v-if="hoverPixel.angle_deg !== null">
                  · 波束角={{ hoverPixel.angle_deg.toFixed ? hoverPixel.angle_deg.toFixed(2) : hoverPixel.angle_deg }}°
                </span>
                <span v-if="hoverPixel.processed === null">（扇形映射后该坐标不对应原始像素）</span>
              </template>
              <template v-else>将鼠标移到图像上查看算法处理后的数值</template>
            </div>
          </div>
        </div>

        <!-- ------------------------------ 帧元数据 ------------------------------ -->
        <div class="viewer-panel" style="margin-top: 12px">
          <div class="viewer-panel-title">
            <span>当前帧元数据</span>
            <span>快捷键：空格播放/暂停 · ←/→ 单帧 · Home/End 首末帧</span>
          </div>
          <div class="viewer-meta">
            <div>帧号(1基): <b>{{ index + 1 }}</b></div>
            <div>ping: <b>{{ meta.ping_num }}</b></div>
            <div>波束数: <b>{{ meta.nBeam }}</b></div>
            <div>采样数: <b>{{ meta.nRange }}</b></div>
            <div>量程: <b>{{ formatNumber(meta.range_m, 2) }} m</b></div>
            <div>开角: <b>{{ formatNumber(meta.open_angle_deg, 2) }}°</b></div>
            <div>频率: <b>{{ formatNumber(meta.frequency, 2) }} kHz</b></div>
            <div>脉宽: <b>{{ formatNumber(meta.pluse_width, 2) }} us</b></div>
            <div>带宽: <b>{{ formatNumber(meta.band_width, 2) }} kHz</b></div>
            <div>水深: <b>{{ formatNumber(meta.water_depth, 2) }} m</b></div>
            <div>高模式: <b>{{ meta.high_mode ? '是' : '否' }}</b></div>
            <div>时间戳: <b>{{ formatTime(meta.time_stamp) }}</b></div>
          </div>
          <div class="dap-muted" style="margin-top: 6px">
            预取 {{ bufferSize }} 帧（缓存命中 {{ cacheHits }} 次）· 批量接口一次取 {{ batchCount }} 帧
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import {
  ArrowLeft,
  ArrowRight,
  DArrowLeft,
  RefreshRight,
  VideoPause,
  VideoPlay,
} from '@element-plus/icons-vue'
import api from '../api/client'
import { PROTOCOL_TEXT, formatNumber, formatTime } from '../utils/format'

const props = defineProps({
  fileId: { type: [Number, String], required: true },
  info: { type: Object, required: true },
})
const emit = defineEmits(['stats'])

const algorithms = ref(props.info.algorithms || [])
const colormaps = ref(props.info.colormaps || ['gray'])
const frameCount = ref(props.info.frame_count || 0)
const supportsSectorFan = ref(Boolean(props.info.capabilities?.sector_fan))
const playback = props.info.playback || {}
const BASE_INTERVAL = playback.base_interval_ms ?? 180
const MIN_INTERVAL = playback.min_interval_ms ?? 40
const batchCount = Math.min(playback.batch_count ?? 32, playback.batch_max_count ?? 32)

const speedOptions = [0.25, 0.5, 1, 2, 4, 8]
const speed = ref(1)
const enabledAlgos = ref([])
const colormap = ref(props.info.default_colormap || (props.info.protocol === 'norbit' ? 'viridis' : 'jet'))
const brightness = ref(0)
const contrast = ref(100)
const sectorFan = ref(false)

const index = ref(0)
const jumpIndex = ref(1)
const playing = ref(false)
const loading = ref(false)
const current = ref(null)
const hoverPixel = ref(null)
const hoverPanel = ref('color')
const hoverSize = ref(null)
const statusText = ref('就绪')
const cacheHits = ref(0)
const bufferSize = ref(0)

const buffer = new Map()
let timer = null
let hoverTimer = null
let fetchToken = 0
let paramsSignature = ''

// 与桌面版一致：step = max(1, int(speed))，interval = max(40, int(180 / speed))
const stepSize = computed(() => Math.max(1, Math.trunc(speed.value)))
const intervalMs = computed(() => Math.max(MIN_INTERVAL, Math.trunc(BASE_INTERVAL / speed.value)))

const protocolLabel = computed(() => PROTOCOL_TEXT[props.info.protocol] || props.info.protocol || 'unknown')
const meta = computed(() => current.value?.meta || props.info.first_frame_meta || {})
const shapeText = computed(() => {
  const shape = (current.value?.shapes && current.value.shapes.gray) || props.info.image_shape || []
  return shape.length === 2 ? `画布 ${shape[1]} × ${shape[0]}` : ''
})
const colorShapeText = computed(() => {
  const shape = (current.value?.shapes && current.value.shapes.color) || props.info.image_shape || []
  const gray = (current.value?.shapes && current.value.shapes.gray) || props.info.image_shape || []
  if (shape.length !== 2) return ''
  const sameAsGray = shape[0] === gray[0] && shape[1] === gray[1]
  // 开启扇形映射后伪彩栏会被重采样成更宽的画布（与桌面版一致）
  return sameAsGray ? `画布 ${shape[1]} × ${shape[0]}` : `画布 ${shape[1]} × ${shape[0]}（扇形映射）`
})
const canvasStyle = computed(() => ({ minHeight: '180px' }))
const crosshairStyle = computed(() => {
  // 十字准星必须按**所在栏**的画布尺寸定位（两栏尺寸可能不同）
  const shape = hoverSize.value || props.info.image_shape || []
  if (shape.length !== 2 || !hoverPixel.value) return { display: 'none' }
  const [rows, cols] = shape
  return {
    left: `${((hoverPixel.value.x + 0.5) / cols) * 100}%`,
    top: `${((hoverPixel.value.y + 0.5) / rows) * 100}%`,
  }
})

function currentParams() {
  return {
    algos: enabledAlgos.value.join(','),
    colormap: colormap.value,
    brightness: brightness.value,
    contrast: contrast.value / 100,
    sector_fan: sectorFan.value ? 'true' : 'false',
  }
}

function signature(params) {
  return [params.algos, params.colormap, params.brightness, params.contrast, params.sector_fan].join('|')
}

function clearBuffer() {
  buffer.clear()
  bufferSize.value = 0
}

function trimBuffer() {
  // 播放缓冲：保留当前帧附近最多 64 帧，避免长时间播放内存增长
  bufferSize.value = buffer.size
  if (buffer.size <= 64) return
  const keepFrom = Math.max(0, index.value - 16)
  for (const key of [...buffer.keys()]) {
    if (key < keepFrom) buffer.delete(key)
  }
  bufferSize.value = buffer.size
}

async function fetchBatch(start, step) {
  const params = currentParams()
  const token = ++fetchToken
  loading.value = true
  try {
    const response = await api.sonarFrames(props.fileId, {
      ...params,
      start,
      count: batchCount,
      step,
    })
    if (token !== fetchToken) return null
    response.frames.forEach((frame) => buffer.set(frame.index, frame))
    bufferSize.value = buffer.size
    return response.frames
  } finally {
    if (token === fetchToken) loading.value = false
  }
}

async function showFrame(target, { prefetchStep = null } = {}) {
  const clamped = Math.max(0, Math.min(target, frameCount.value - 1))
  index.value = clamped
  jumpIndex.value = clamped + 1

  if (buffer.has(clamped)) {
    cacheHits.value += 1
    current.value = buffer.get(clamped)
    statusText.value = `帧 ${clamped + 1}/${frameCount.value}（缓冲命中）`
    emit('stats', { cacheHits: cacheHits.value })
    return
  }

  const step = prefetchStep ?? stepSize.value
  current.value = null
  try {
    const frames = await fetchBatch(clamped, step)
    if (!frames) return
    const frame = buffer.get(clamped) || frames[0]
    current.value = frame || null
    statusText.value = `帧 ${clamped + 1}/${frameCount.value}（已预取 ${frames.length} 帧）`
    emit('stats', { cacheHits: cacheHits.value })
  } catch (error) {
    pause()
    statusText.value = `取帧失败：${error.message}`
    ElMessage.error(error.message)
  }
  trimBuffer()
}

function onParamsChanged() {
  const params = currentParams()
  paramsSignature = signature(params)
  clearBuffer()
  fetchToken += 1
  showFrame(index.value)
}

function onAdjustInput() {
  // 拖动过程中只更新数值，避免每个像素都请求；松手时由 @change 触发实际刷新
  paramsSignature = signature(currentParams())
}

function resetDisplay() {
  brightness.value = 0
  contrast.value = 100
  colormap.value = props.info.default_colormap || (props.info.protocol === 'norbit' ? 'viridis' : 'jet')
  sectorFan.value = false
  enabledAlgos.value = []
  onParamsChanged()
}

function clearAlgorithms() {
  enabledAlgos.value = []
  onParamsChanged()
}

function stepFrame(delta) {
  pause()
  showFrame(index.value + delta)
}

function onJump() {
  pause()
  showFrame(Math.max(0, Math.min((jumpIndex.value || 1) - 1, frameCount.value - 1)))
}

function play() {
  if (playing.value || frameCount.value <= 0) return
  playing.value = true
  statusText.value = '播放中'
  scheduleTick()
}

function pause() {
  if (!playing.value) return
  playing.value = false
  statusText.value = `已暂停（帧 ${index.value + 1}）`
  if (timer) {
    clearTimeout(timer)
    timer = null
  }
}

function stop() {
  pause()
  showFrame(0)
  statusText.value = '已停止并回到首帧'
}

function scheduleTick() {
  if (timer) clearTimeout(timer)
  timer = setTimeout(async () => {
    if (!playing.value) return
    const next = index.value + stepSize.value
    if (next >= frameCount.value) {
      await showFrame(frameCount.value - 1)
      pause()
      statusText.value = '已播放到最后一帧'
      return
    }
    await showFrame(next, { prefetchStep: stepSize.value })
    if (playing.value) scheduleTick()
  }, intervalMs.value)
}

function onHover(event, panel) {
  const image = event.currentTarget
  const rect = image.getBoundingClientRect()
  if (!rect.width || !rect.height) return
  const naturalWidth = image.naturalWidth || image.width
  const naturalHeight = image.naturalHeight || image.height
  const x = Math.floor(((event.clientX - rect.left) / rect.width) * naturalWidth)
  const y = Math.floor(((event.clientY - rect.top) / rect.height) * naturalHeight)
  hoverPanel.value = panel
  hoverSize.value = [naturalHeight, naturalWidth]
  hoverPixel.value = { x, y, raw: '…', processed: '…', angle_deg: null }

  if (hoverTimer) clearTimeout(hoverTimer)
  hoverTimer = setTimeout(async () => {
    try {
      // 按栏上报坐标：伪彩栏的坐标系可能与灰度栏不同
      const result = await api.sonarPixel(props.fileId, {
        ...currentParams(),
        index: index.value,
        x,
        y,
        panel,
      })
      hoverPixel.value = result
      if (result.panel_shape) hoverSize.value = result.panel_shape
    } catch (error) {
      hoverPixel.value = null
    }
  }, 90)
}

function clearHover() {
  if (hoverTimer) clearTimeout(hoverTimer)
  hoverPixel.value = null
  hoverSize.value = null
}

function onKeydown(event) {
  const tag = (event.target?.tagName || '').toLowerCase()
  if (['input', 'textarea'].includes(tag)) return
  if (event.code === 'Space') {
    event.preventDefault()
    playing.value ? pause() : play()
  } else if (event.code === 'ArrowLeft') {
    event.preventDefault()
    stepFrame(-stepSize.value)
  } else if (event.code === 'ArrowRight') {
    event.preventDefault()
    stepFrame(stepSize.value)
  } else if (event.code === 'Home') {
    event.preventDefault()
    stepFrame(-frameCount.value)
  } else if (event.code === 'End') {
    event.preventDefault()
    showFrame(frameCount.value - 1)
  }
}

watch(
  () => props.fileId,
  () => {
    pause()
    clearBuffer()
    frameCount.value = props.info.frame_count || 0
    showFrame(0)
  },
)

onMounted(async () => {
  window.addEventListener('keydown', onKeydown)
  if (!frameCount.value) {
    try {
      const info = await api.sonarInfo(props.fileId)
      frameCount.value = info.frame_count || 0
    } catch (error) {
      /* 忽略 */
    }
  }
  if (frameCount.value > 0) await showFrame(0)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeydown)
  if (timer) clearTimeout(timer)
  if (hoverTimer) clearTimeout(hoverTimer)
})
</script>
