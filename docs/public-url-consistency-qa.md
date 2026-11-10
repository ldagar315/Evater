# Public URL consistency

10 November 2026, before deployment. About, blog index and ten articles are served as directory indexes at trailing-slash URLs. The previous sitemap, canonical tags and structured data advertised redirecting no-slash URLs. This is a consistency fix, not evidence of lost content, traffic or indexing. Search Console's Page with redirect exclusion describes the redirect URL, not necessarily its destination.

## Scope

- Align public canonicals, og:url, JSON-LD, sitemap and internal links with the existing hosting convention.
- Preserve app/auth paths, query strings, About fragments and both hosting rule sources. Existing no-slash inbound URLs keep their hosting redirects.
- About remains recognized as a public route and selected nav item at either spelling.
- No hosting settings, Search Console validation requests, database writes, generation calls or merge.

## Checks

- `npx tsc --noEmit`: pass.
- `npm test`: 16 tests pass, including three URL-helper cases.
- `npm run build`: pass, including prerender output.
- `python3 check-app-route-rules.py`: 19 static app paths preserve their fallbacks in both rule sources, unknown fallback remains404.
- `node qa-public-urls.mjs` after build: 13 sitemap URLs unique, all directly200 on a directory-index server, each matching canonical and og:url; public links aligned. About trailing-slash nav and hash, desktop blog, mobile article and horizontal overflow checked. External APIs blocked; public images allowed. Screenshots manually inspected for readability and layout.
- Read-only production baseline: all 13 trailing-slash targets already return200 without redirect, recorded in `public-url-live-baseline.json`. Production still has the old canonical tags until this PR is merged and deployed.

## After Lakshay merges

Check all live sitemap targets again: direct200, matching canonical/og:url/structured data, old no-slash redirects retained, About hash/category links and app direct-entry routes preserved. Inspect deployed pages. Google recrawl and report refresh are asynchronous; do not label exclusions fixed merely because deployment passes.

Google reference: https://support.google.com/webmasters/answer/7440203#page_with_redirect
