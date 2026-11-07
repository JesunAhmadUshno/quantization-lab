import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// The site reads ../level-*/results_*.json at build time, so every number on the page is generated, never typed.
export default defineConfig({
  base: '/quantization-lab/',
  plugins: [react()],
  server: { fs: { allow: ['..'] } },
});
