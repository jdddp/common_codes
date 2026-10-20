<template>
  <el-container style="height: 100%">
    <el-header class="app-header">
      <div class="brand">
        <span class="brand-dot" />
        内部数据资产管理平台
      </div>
      <el-menu
        mode="horizontal"
        :default-active="activeMenu"
        :ellipsis="false"
        router
        class="app-menu"
      >
        <el-menu-item index="/files">文件检索</el-menu-item>
        <el-menu-item index="/admin/attributes">属性与模板</el-menu-item>
      </el-menu>
      <div class="app-header-right">
        <el-tag type="info" effect="plain" size="small">内网单机 · 免登录</el-tag>
      </div>
    </el-header>
    <el-main style="padding: 0; overflow: auto">
      <router-view />
    </el-main>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const activeMenu = computed(() => {
  const path = route.path
  if (path.startsWith('/admin')) return '/admin/attributes'
  return '/files'
})
</script>

<style scoped>
.app-header {
  display: flex;
  align-items: center;
  gap: 24px;
  background: #fff;
  border-bottom: 1px solid #e6e9f0;
  height: 56px;
  padding: 0 20px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: 16px;
  white-space: nowrap;
}

.brand-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: linear-gradient(135deg, #3f8cff, #17c3b2);
}

.app-menu {
  border-bottom: none !important;
  flex: 1;
}

.app-header-right {
  white-space: nowrap;
}
</style>
