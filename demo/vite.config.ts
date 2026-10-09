import { resolve } from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      webgazer: resolve(__dirname, '../src/index.ts'),
      'gaze-model-source': resolve(__dirname, '../src/gazeModelLocal.ts')
    }
  },
  server: {
    port: 3000,
    open: true
  }
});
