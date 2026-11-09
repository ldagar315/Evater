# Gurugram school suggestions

Source snapshot: 9 November 2026. This is a partial directory, not a verified register of every active school.

- CBSE SARAS affiliated-school search: https://saras.cbse.gov.in/saras/AffiliatedList/ListOfSchdirReport . State HARYANA (5), district GURUGRAM (05): 288 returned affiliation records. Each saved entry retains its affiliation number and the exact details-page URL returned by SARAS.
- Gurugram district schools directory: https://gurugram.gov.in/public-utility-category/schools/ . Pages 1 through 52 returned names; the next page returned none. 467 unique raw names, reduced to466 added suggestions after normalized exact-name deduplication against all entries. Each entry retains the page where the name was observed.
- Total754 suggestions. Different campuses remain separate. Normalization only collapses exact names after case/punctuation and Gurgaon/Gurugram normalization; it does not prove two differently named records are different institutions. No fuzzy merge, affiliation-status guarantee, current-opening guarantee, or exhaustive coverage claim.

Only school names and campus addresses are bundled. Principal names, phone numbers, and email addresses are not included. CBSE source spelling and casing are preserved. CBSE labels include source addresses to separate campuses (for example, Euro International sectors10,45,37D,84,109 and51). District-only entries may lack precise campus addresses.

## Storage and existing profiles

The existing `Users.school` text column stores a selected canonical label. No schema migration or rewriting of old profiles. An existing unmatched name stays in the custom field. Other is always available, including when there are matches. Typing an unselected search query does not save a new variant by accident. Custom spelling is not silently corrected.

This improves consistency for new selections; it does not retroactively merge existing school leaderboard groups or guarantee alias resolution. Selecting the same school through a differently named district record can still produce a distinct stored label. Review/deduplication beyond exact names needs care because similarly named campuses must not be merged.

## Refresh

`python3 scripts/update-gurugram-schools.py` (requires requests and beautifulsoup4). The script uses the source's own SARAS form and district pagination, and fails if SARAS yields no entries. It reads public sources only and writes no database. Review the resulting diff, source counts, campus distinctions and UI provenance date before committing. Refreshes can change names, addresses and coverage. Stable CBSE affiliation IDs and hashed district names avoid ordinal churn.

The `--cbse-html` and `--district-json` options allow offline reproduction from saved source responses. Do not commit raw responses containing unrelated contact details.
