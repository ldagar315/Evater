import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import prerender from '@prerenderer/rollup-plugin';
import PuppeteerRenderer from '@prerenderer/renderer-puppeteer';

// Public, crawlable routes that get static HTML at build time. App routes stay client-rendered.
const PRERENDER_ROUTES = [
  '/',
  '/about',
  '/papers',
  '/papers/class-8-maths',
  '/papers/class-8-science',
  '/blog',
  '/blog/class-8-maths-a-square-and-a-cube-study-resources',
  '/blog/class-8-maths-fractions-in-disguise-study-resources',
  '/blog/class-8-science-how-nature-works-in-harmony-study-resources',
  '/blog/class-8-maths-a-story-of-numbers-study-resources',
  '/blog/class-8-maths-we-distribute-yet-things-multiply-study-resources',
  '/blog/class-8-science-the-invisible-living-world-beyond-our-naked-eye-study-resources',
  '/blog/class-8-maths-power-play-study-resources',
  '/blog/class-8-maths-proportional-reasoning-1-study-resources',
  '/blog/class-8-science-exploring-forces-study-resources',
  '/blog/ncert-class-8-maths-power-play-study-guide',
  '/blog/ncert-class-8-maths-area-study-guide',
  '/blog/ncert-class-8-science-keeping-time-skies-study-guide',
  '/blog/ncert-class-8-science-nature-ecosystems-harmony-study-guide',
  '/blog/ncert-class-8-science-earth-life-sustaining-planet-study-guide',

  '/blog/ncert-class-8-science-elements-compounds-mixtures-study-guide',
  '/blog/ncert-class-8-science-solutes-solvents-solutions-density-study-guide',

  '/blog/ncert-class-8-science-electricity-magnetic-heating-effects-study-guide',
  '/blog/ncert-class-8-science-pressure-winds-storms-cyclones-study-guide',
  '/blog/ncert-class-8-science-particulate-nature-of-matter-study-guide',

  '/blog/ncert-class-8-science-investigative-world-study-guide',
  '/blog/ncert-class-8-science-invisible-living-world-study-guide',
  '/blog/ncert-class-8-science-health-ultimate-treasure-study-guide',

  '/blog/cbse-class-8-science-study-plan-30-day-ncert-revision',
  '/blog/active-recall-class-6-10-exams',
  '/blog/ncert-class-8-science-prepare-chapter-tests',
  '/blog/study-difficult-science-chapters-without-rote-memorisation',
  '/blog/practice-tests-vs-revision-exam-scores',
  '/blog/how-students-in-middle-school-should-learn',
  '/blog/ncert-class-8-maths-baudhayana-pythagoras-theorem',
  '/blog/ncert-class-8-science-light-mirrors-lenses-study-guide',
  '/blog/ncert-class-8-maths-square-and-cube-study-guide',
  '/blog/ncert-class-8-science-exploring-forces-study-guide',

  // Paper detail pages: one crawlable route per catalog paper.
  '/papers/class-8-maths-a-square-and-a-cube-sample-paper',
  '/papers/class-8-maths-power-play-sample-paper',
  '/papers/class-8-maths-a-story-of-numbers-sample-paper',
  '/papers/class-8-maths-quadrilaterals-sample-paper',
  '/papers/class-8-maths-number-play-sample-paper',
  '/papers/class-8-maths-we-distribute-yet-things-multiply-sample-paper',
  '/papers/class-8-maths-proportional-reasoning-1-sample-paper',
  '/papers/class-8-maths-fractions-in-disguise-sample-paper',
  '/papers/class-8-maths-pythagoras-sample-paper',
  '/papers/class-8-maths-proportional-reasoning-2-sample-paper',
  '/papers/class-8-maths-exploring-some-geometric-themes-sample-paper',
  '/papers/class-8-maths-tales-by-dots-and-lines-sample-paper',
  '/papers/class-8-maths-algebra-play-sample-paper',
  '/papers/class-8-maths-area-sample-paper',
  '/papers/class-8-science-exploring-the-investigative-world-of-science-sample-paper',
  '/papers/class-8-science-the-invisible-living-world-beyond-our-naked-eye-sample-paper',
  '/papers/class-8-science-health-the-ultimate-treasure-sample-paper',
  '/papers/class-8-science-electricity-sample-paper',
  '/papers/class-8-science-exploring-forces-sample-paper',
  '/papers/class-8-science-pressure-winds-storms-and-cyclones-sample-paper',
  '/papers/class-8-science-particulate-nature-of-matter-sample-paper',
  '/papers/class-8-science-nature-of-matter-elements-compounds-and-mixtures-sample-paper',
  '/papers/class-8-science-the-amazing-world-of-solutes-solvents-and-solutions-sample-paper',
  '/papers/class-8-science-light-sample-paper',
  '/papers/class-8-science-keeping-time-with-the-skies-sample-paper',
  '/papers/class-8-science-how-nature-works-in-harmony-sample-paper',
  '/papers/class-8-science-our-home-earth-a-unique-life-sustaining-planet-sample-paper',
  '/papers/class-8-maths-mixed-chapter-practice-paper',
  '/papers/class-8-science-mixed-chapter-practice-paper',
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
                maxConcurrentRoutes: 4,
                skipThirdPartyRequests: true,
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
