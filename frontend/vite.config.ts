import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const modalApiUrl = (
    env.VITE_MODAL_API_URL || 'https://ldagar315--evater-v1-wrapper.modal.run'
  ).replace(/\/+$/, '')

  return {
    plugins: [react()],
    optimizeDeps: {
      exclude: ['lucide-react'],
    },
    server: {
      proxy: {
        '/api/external': {
          target: modalApiUrl,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/external/, ''),
          secure: true,
        },
      },
    },
  }
})
