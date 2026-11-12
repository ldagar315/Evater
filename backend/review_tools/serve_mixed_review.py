"""Run with explicit local inputs; no environment secrets or database access."""
import argparse
import json
from pathlib import Path
from fastapi.responses import FileResponse
from review_tools.mixed_http_review import create_app, synthetic_inputs, NotReady

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    synthetic, snapshot = synthetic_inputs()
    papers = {**inputs['papers'], 'synthetic': synthetic, 'broken': synthetic}
    snapshots = {**inputs['snapshots'], 'synthetic': snapshot, 'broken': snapshot}
    def loader(paper_id):
        from review_tools.mixed_http_review import build_payload
        if paper_id == 'broken':
            raise RuntimeError('synthetic injected loader failure')
        return build_payload(papers[paper_id], snapshots[paper_id])
    app = create_app(papers, snapshots, loader=loader)
    assets = Path(__file__).resolve().parents[2] / 'frontend/review-tools'
    @app.get('/')
    def index():
        return FileResponse(assets / 'index.html')
    @app.get('/assets/review.js')
    def javascript():
        return FileResponse(assets / 'review.js', media_type='text/javascript')
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')
if __name__ == '__main__':
    main()
