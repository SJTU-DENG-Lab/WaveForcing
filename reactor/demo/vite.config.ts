import { defineConfig } from "vite";
export default defineConfig({
  optimizeDeps: {
    exclude: ["@reactor-team/js-sdk"],
    include: ["react", "react/jsx-runtime", "react-dom/client", "awaitqueue"],
  },
});
