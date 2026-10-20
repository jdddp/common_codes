import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发时：前端 5173，后端 8000；`npm run build` 产物落到 dist/，
// 由后端 FastAPI 直接托管（无需 nginx）。
export default defineConfig({
  plugins: [vue()],
  base: './',
  server: {
    port: 5173,
    host: '127.0.0.1',
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 1500,
  },
})
