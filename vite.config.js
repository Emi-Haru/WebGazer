import { copyFileSync, mkdirSync } from 'fs';
import { resolve } from 'path';
import { defineConfig } from 'vite';
import banner from 'vite-plugin-banner';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const bannerString = `
 WebGazer.js: Democratizing Webcam Eye Tracking on the Browser
 Copyright (c) 2016, Brown WebGazer Team
 Licensed under GPLv3. Companies with a valuation of less than $1M can use WebGazer.js under LGPLv3.
 `;

export default defineConfig({
  resolve: {
    alias: {
      'gaze-model-source': resolve(__dirname, 'src/gazeModelPackaged.ts')
    }
  },
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'webgazer',
      formats: ['es', 'cjs'],
      fileName: (format) => `webgazer.${format}.js`
    },
    sourcemap: true,
    outDir: 'dist',
    emptyOutDir: false,
    rollupOptions: {
      output: {
        inlineDynamicImports: true
      }
    },
    minify: 'esbuild'
  },
  plugins: [
    banner(bannerString),
    {
      name: 'copy-gaze-model',
      closeBundle () {
        const destDir = resolve(__dirname, 'dist/models');
        mkdirSync(destDir, { recursive: true });
        copyFileSync(
          resolve(__dirname, 'src/models/gaze_corner_frame.bin'),
          resolve(destDir, 'gaze_corner_frame.bin')
        );
      }
    },
    {
      name: 'emit-types',
      async closeBundle () {
        try {
          await execAsync('tsc --emitDeclarationOnly --declaration --outDir dist/types');
        } catch (error) {
          console.error('Error in closeBundle:', error);
        }
      }
    }
  ]
});
