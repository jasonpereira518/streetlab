export const RELEASES_URL = 'https://github.com/jasonpereira518/streetlab/releases';

/** Shown only in VITE_DEMO builds: the hosted demo runs the in-process mock,
 * not the Python simulator, and must say so. */
export function DemoBanner() {
  return (
    <div className="demo-banner" role="note">
      Live demo: scripted data in your browser, not the real Python simulator.{' '}
      <a href={RELEASES_URL} target="_blank" rel="noreferrer">
        Get the full app
      </a>
    </div>
  );
}
