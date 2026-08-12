import ReactDOM from 'react-dom/client';
import App from './App';
import { installThemeVars } from './ui/theme';
import './styles.css';

installThemeVars();

// No StrictMode: the WebGPU renderer owns an async GPU device and a persistent
// animation loop, and double-invoked effects would tear it down mid-init.
ReactDOM.createRoot(document.getElementById('root')!).render(<App />);
