"use client";

interface Step {
  number: string;
  label: string;
  description: string;
}

export default function InfoCards({ steps }: { steps: Step[] }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 border border-border rounded-lg bg-panel divide-y sm:divide-y-0 sm:divide-x divide-border mb-6">
      {steps.map((step) => (
        <div key={step.number} className="px-6 py-6 text-center">
          <p className="font-mono text-3xl font-bold text-accent mb-1">{step.number}</p>
          <p className="font-mono text-xs text-muted tracking-widest mb-2">{step.label}</p>
          <p className="text-sm text-text">{step.description}</p>
        </div>
      ))}
    </div>
  );
}
