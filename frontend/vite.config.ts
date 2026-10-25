import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` (port 5173) proxies the API to the FastAPI server on port 8000.
// Prod: `npm run build` writes frontend/dist, which FastAPI serves itself (one process).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/download": "http://localhost:8000",
    },
  },
});
