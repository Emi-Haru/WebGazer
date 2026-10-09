import { resolve } from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ command }) => ({
  // Dev stays at /. The Pages site is served from /WebGazer/.
  base: command === 'build' ? '/WebGazer/' : '/',
  plugins: [react()],
  resolve: {
    alias: {
      webgazer: resolve(__dirname, '../src/index.ts'),
      'gaze-model-source': command === 'build'
        ? resolve(__dirname, 'src/gazeModelRemote.ts')
        : resolve(__dirname, '../src/gazeModelLocal.ts')
    }
  },
  server: {
    port: 3001,
    open: true
  },
  optimizeDeps: {
    include: [
      '@tensorflow/tfjs',
      '@mediapipe/tasks-vision'
    ]
  }
}));
