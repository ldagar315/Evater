import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import prerender from '@prerenderer/rollup-plugin';
import PuppeteerRenderer from '@prerenderer/renderer-puppeteer';

// Public, crawlable routes that get static HTML at build time. App routes stay client-rendered.
const PRERENDER_ROUTES = [
  '/',
  '/about',
  // Sample papers (kebab-case chapter slugs; same slugify rule as the backend
  // sample_papers_router). Hubs + Class 8 chapters, added with the routes PR.
  '/sample-papers',
  '/sample-papers/class-8',
  '/sample-papers/class-8/science',
  '/sample-papers/class-8/mathematics',
  '/sample-papers/class-8/science/exploring-the-investigative-world-of-science',
  '/sample-papers/class-8/science/the-invisible-living-world',
  '/sample-papers/class-8/science/health-the-ultimate-treasure',
  '/sample-papers/class-8/science/electricity-magnetic-and-heating-effects',
  '/sample-papers/class-8/science/exploring-forces',
  '/sample-papers/class-8/science/pressure-winds-storms-and-cyclones',
  '/sample-papers/class-8/science/particulate-nature-of-matter',
  '/sample-papers/class-8/science/nature-of-matter-elements-compounds-and-mixtures',
  '/sample-papers/class-8/science/the-amazing-world-of-solutes-solvents-and-solutions',
  '/sample-papers/class-8/science/light-mirrors-and-lenses',
  '/sample-papers/class-8/science/keeping-time-with-the-skies',
  '/sample-papers/class-8/science/how-nature-works-in-harmony',
  '/sample-papers/class-8/science/our-home-earth-a-unique-life-sustaining-planet',
  '/sample-papers/class-8/mathematics/a-square-and-a-cube',
  '/sample-papers/class-8/mathematics/power-play',
  '/sample-papers/class-8/mathematics/a-story-of-numbers',
  '/sample-papers/class-8/mathematics/quadrilaterals',
  '/sample-papers/class-8/mathematics/number-play',
  '/sample-papers/class-8/mathematics/we-distribute-yet-things-multiply',
  '/sample-papers/class-8/mathematics/proportional-reasoning-1',
  '/sample-papers/class-8/mathematics/fractions-in-disguise',
  '/sample-papers/class-8/mathematics/the-baudhayana-pythagoras-theorem',
  '/sample-papers/class-8/mathematics/proportional-reasoning-2',
  '/sample-papers/class-8/mathematics/exploring-some-geometric-themes',
  '/sample-papers/class-8/mathematics/tales-by-dots-and-lines',
  '/sample-papers/class-8/mathematics/algebra-play',
  '/sample-papers/class-8/mathematics/area',
  '/blog',
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
                renderAfterTime: 12000,  // generous wait so baked pages are never mid-fetch "Loading..." states; sample pages are noindex until data renders
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
