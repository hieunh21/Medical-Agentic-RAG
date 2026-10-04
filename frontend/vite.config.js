import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api sang backend :8000 để frontend gọi đường dẫn tương đối — không cần
// cấu hình URL theo môi trường, và tránh CORS khi build production chung domain.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.API_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
