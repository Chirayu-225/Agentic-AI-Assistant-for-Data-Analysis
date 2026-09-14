// plotly.js-dist-min ships no type declarations. Only the one function
// lib/pdfReport.ts actually uses (Plotly.toImage, for exporting a
// rendered chart to a PNG for the PDF report) is declared here — not a
// full surface, since nothing else in this app imports this package
// directly (react-plotly.js bundles its own copy of plotly.js
// internally for rendering).
declare module "plotly.js-dist-min" {
  interface ToImageOptions {
    format?: "png" | "jpeg" | "webp" | "svg";
    width?: number;
    height?: number;
  }

  interface PlotlyStatic {
    toImage(graphDiv: HTMLElement, opts?: ToImageOptions): Promise<string>;
  }

  const Plotly: PlotlyStatic;
  export default Plotly;
}
