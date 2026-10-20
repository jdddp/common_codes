import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/files' },
  {
    path: '/files',
    name: 'files',
    component: () => import('../views/SearchView.vue'),
    meta: { title: '文件检索' },
  },
  {
    path: '/files/:id',
    name: 'file-detail',
    component: () => import('../views/FileDetailView.vue'),
    meta: { title: '文件详情 / 在线查看' },
  },
  {
    path: '/admin/attributes',
    name: 'attributes',
    component: () => import('../views/AttrAdminView.vue'),
    meta: { title: '属性与模板' },
  },
]

// 使用 hash 模式：内网直接以静态文件方式部署时无需后端 rewrite 规则
const router = createRouter({ history: createWebHashHistory(), routes })

router.afterEach((to) => {
  document.title = to.meta?.title ? `${to.meta.title} · 数据资产管理平台` : '数据资产管理平台'
})

export default router
