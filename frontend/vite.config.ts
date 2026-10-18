import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import prerender from '@prerenderer/rollup-plugin';
import PuppeteerRenderer from '@prerenderer/renderer-puppeteer';

// Public, crawlable routes that get static HTML at build time. App routes stay client-rendered.
const PRERENDER_ROUTES = [
  '/',
  '/about',
  '/blog',
  '/blog/cbse-class-8-science-study-plan-30-day-ncert-revision',
  '/blog/active-recall-class-6-10-exams',
  '/blog/ncert-class-8-science-prepare-chapter-tests',
  '/blog/study-difficult-science-chapters-without-rote-memorisation',
  '/blog/practice-tests-vs-revision-exam-scores',
  '/blog/how-students-in-middle-school-should-learn',
];

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const modalApiUrl = (
    env.VITE_MODAL_API_URL || 'https://ldagar315--evater-v1-wrapper.modal.run'
  ).replace(/\/+$/, '')

  return {
    plugins: [
      react(),
      // Set PRERENDER=false to skip (e.g. if the build environment has no Chromium).
      ...(mode === 'production' && process.env.PRERENDER !== 'false'
        ? [
            prerender({
              routes: PRERENDER_ROUTES,
              renderer: new PuppeteerRenderer({
                renderAfterTime: 3000,
                launchOptions: { args: ['--no-sandbox', '--disable-setuid-sandbox'] },
              }),
            }),
          ]
        : []),
    ],
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

