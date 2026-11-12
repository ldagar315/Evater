// Explicit local harness entry. Not registered in App or a public route.
import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {MixedSamplePaperReview, MixedReviewPaper} from '../src/components/MixedSamplePaperReview';

type State = {loading: boolean; paper?: MixedReviewPaper; issues?: string[]};
function ReviewHarness() {
  const [id, setId] = useState('synthetic');
  const [state, setState] = useState<State>({loading: true});
  useEffect(() => {
    const controller = new AbortController();
    setState({loading: true}); // A stale successful paper is never kept after a failed selection.
    fetch(`/review/mixed-sample-papers/${encodeURIComponent(id)}`, {signal: controller.signal, cache: 'no-store'})
      .then(async response => {
        const body = await response.json();
        if (controller.signal.aborted) return;
        if (!response.ok) {
          setState({loading: false, issues: body.detail?.issues ?? [`Paper unavailable (HTTP ${response.status}).`]});
          return;
        }
        setState({loading: false, paper: body});
      }).catch(error => {
        if (error.name !== 'AbortError') setState({loading: false, issues: ['Review request failed. No paper loaded.']});
      });
    return () => controller.abort();
  }, [id]);
  return <>
    <header><strong>Local integration review</strong><p>No production connection. Synthetic success is not curriculum readiness.</p>
      <label>Paper <select aria-label="Paper" value={id} onChange={event => setId(event.target.value)}>
        <option value="synthetic">Synthetic positive control</option>
        <option value="light">Held Light paper</option><option value="electricity">Held Electricity paper</option>
        <option value="maths">Held Maths paper</option><option value="unknown">Unknown paper (negative)</option>
        <option value="broken">Loader failure (negative)</option>
      </select></label></header>
    {state.loading && <p role="status">Loading review over HTTP...</p>}
    {state.issues && <section role="alert"><h1>Review unavailable</h1><p>No substitute paper or answer key was loaded.</p><ul>{state.issues.map(issue => <li key={issue}>{issue}</li>)}</ul></section>}
    {state.paper && <MixedSamplePaperReview paper={state.paper}/>}
  </>;
}
createRoot(document.getElementById('root')!).render(<ReviewHarness/>);
