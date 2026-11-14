# Interactive cube-volume model

Adds original 3D projected unit-cube geometry to the existing square/cube article. No external simulation vendor, package, account, billing, backend, auth, or article-body content changes. Side length1-6, visible layer count1-n, camera angle20-70degrees, reset. No autoplay or transition; respects a reduced-motion preference by remaining user-controlled. All sliders have native keyboard operation and visible labels; text and SVG descriptions carry the math without relying on color.

## Run

From frontend, install locked dependencies then:

```
npm test
PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome npm run build
node qa-cube-volume.mjs cube-volume-evidence
npx tsc --noEmit -p tsconfig.app.json
```

The browser script uses an installed Chrome at /usr/bin/google-chrome. It serves the actual build locally and blocks all external requests. It checks initial count, keyboard changes to216cubes, partial layers72cubes, rotation without count change, reset, every side1-6, labels, overflow, arithmetic, and exclusion from an unrelated article at320/390/1280. Screenshots use a taller viewport only to capture the whole model without the site's sticky header covering it.

## Evidence and limits

36browser checks passed, zero browser errors.19unit tests passed, including exact side/layer enumeration and every projected vertex in the viewbox at camera limits. Full production prerender/build passed. Full application TypeScript still fails with74normalized diagnostics, identical to exactmaina2c2f84baseline; the root tsconfig is a reference-only file, so bare tsc--noEmit is not evidence of a clean app typecheck.

This is an interactive projected3D geometric model, not a physics engine or WebGL scene. It models volume only, not forces or optical laws. No claim of fullClass8coverage, concept-test funnel completion, real student journey, or live publication. Those remain separate work. User merges the PR.
