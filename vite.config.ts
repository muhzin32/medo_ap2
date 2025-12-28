import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  server: {
    open: true,
    port: 3000,
    proxy: {
      '/api': {
        target: 'https://medo-ap2.onrender.com', // Production Backend
        changeOrigin: true,
      },
    },
  },

  resolve: {
    alias: {
      src: path.resolve(__dirname, './src'),
    },
  },
});
