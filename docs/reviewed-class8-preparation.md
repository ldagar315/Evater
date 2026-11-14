# Class8 reviewed content preparation

This PR prepares the exact80recommended questions and3selectiveguide bodies,excluding10reservevariants. User steering: tests must be usable through Evater UI; database presence alone is not acceptance. No production data writes, guide activation or agent merge are part of this preparation PR.

## Safe behaviour on merge

- Existing blog catalogue/routes and ordinary test behaviour stay unchanged.
- Reviewed guide preview exists only under Vite development mode at /reviewed-content/lenses, /reviewed-content/power-play and /reviewed-content/invisible-living-world.
- Compile-time guards remove the route, component and pending guide text from production assets. Production direct route returns the existing not-found UI.
- No guide is silently added to blogPosts, sitemap or prerender routes.
- Scoped backend adapter has no network/publish client. It validates the exact reviewed proposal and prepares review-status rows only, regardless of manifest approval claims.
- Existing general100-itempublishvalidator remains byte-for-byte unchanged. It is not padded or bypassed.

## Verified locally

12adaptertestsPASS: exact80manifest/counts/mappings/guidehashes,changedtext/difficulty/IDs/reserve/chapter/selection/guidefailclosed,invalidkey/duplicateoptionrejected byexistingmodel,claimedapprovalcannotpublish.

Full production build with Chrome prerender PASS. Production HTML/JS/JSON/Markdown scan has no pending guide text or review route/component. Browser checks:3guides at320/390/1280,allinternalnavigation,unknownslugfallback,productiondirectrouteexclusion;0pageerrors. External HTTP blocked in local browser QA, real local app assets rendered. No live student session or account writes were tested.

Full app TypeScript command remains nonzero on both exactbase andproposal.54normalizeddiagnostics each,noneadded orremoved. This is not a clean-typecheck claim.

Earlier isolated PostgreSQL14.24 rehearsal using repository-derivedquestion_bankDDL:80reviewrowsinsert/rollback/localcommit/idempotencePASS;0publishedrows. ID-only parent stubs do not prove productionRLS/triggers/schema parity.

## Remaining before real UI-ready publication

1. Resolve the exact publication scope from the user's approval exchange; no10reserveorunrelatedcontent.
2. Re-read current production concept/chapter/version IDs,statuses,counts,schema,privileges and source registry. Snapshot IDs are not a live write preflight.
3. Implement a separately reviewed scoped publication path compatible with80items, preserving approved text and difficulty labels. Do not call the old fixed-distribution gate with falsified labels.
4. Integrate approved student guides into the actual public catalogue only when authorized. Keep PR100 capability and live inventory checks for exact concept CTAs,never chapter fallback under a concept label.
5. After owner merge/deploy and authorized import,verify actualpublishedcountsand keys,openblogthroughEvaterUI,sign-in/profilecontinuity,exactconceptsetup/start/answer/submit/results. Syntheticfixtures are not realaccountpersistenceproof. Stop for unresolved identity/permission at that boundary.

Guide scope: Lenses is a concept guide, Power Play and Invisible Living World are selective chapter guides. This is not full Class8coverage or independentteacher/learnercalibrationcertification.
